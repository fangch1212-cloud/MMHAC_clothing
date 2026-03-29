import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from sklearn.cluster import KMeans
from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
from torch_geometric.nn import HGTConv, Linear
args = parse_args()



class MultiHeadModalityAttention(nn.Module):
    def __init__(self, dim):
        super(MultiHeadModalityAttention, self).__init__()
        # 引入特征映射矩阵和全局上下文向量
        self.proj = nn.Linear(dim, dim)
        self.context_vector = nn.Parameter(torch.randn(dim, 1))
        nn.init.xavier_uniform_(self.context_vector.data.unsqueeze(0))

    def forward(self, embeddings_list):
        if not embeddings_list:
            return None, None

        # [N, M, D]
        stack = torch.stack(embeddings_list, dim=1)

        # 统一投影到共享语义空间
        proj_stack = torch.tanh(self.proj(stack))

        # 计算注意力得分
        raw_scores = torch.matmul(proj_stack, self.context_vector)  # [N, M, 1]
        weights = F.softmax(raw_scores, dim=1)

        output = torch.sum(stack * weights, dim=1)
        return output, weights

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(args.UI_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings


class LightHeteroConv(nn.Module):
    """
    轻量级异构图卷积 (Light-HeteroConv)
    完全抛弃 HGT 的密集特征变换矩阵 (W)，仅使用稀疏矩阵乘法进行异构图的结构和语义传播。
    同时自动处理带有相似度权重的边。
    """

    def __init__(self, num_layers, n_users, n_items):
        super(LightHeteroConv, self).__init__()
        self.layers = num_layers
        self.n_users = n_users
        self.n_items = n_items

    def forward(self, u_emb, i_emb, ui_graph, img_adj, txt_adj):
        u_embs = [u_emb]
        i_embs = [i_emb]

        curr_u = u_emb
        curr_i = i_emb

        for _ in range(self.layers):
            # 1. UI 协同过滤交互图传播
            ego = torch.cat([curr_u, curr_i], dim=0)
            cf_next = torch.sparse.mm(ui_graph, ego)
            next_u = cf_next[:self.n_users]
            next_i_cf = cf_next[self.n_users:]

            # 2. Item-Item 模态相似度图传播 (带有对称归一化权重)
            next_i_v = torch.sparse.mm(img_adj, curr_i) if img_adj is not None else 0
            next_i_t = torch.sparse.mm(txt_adj, curr_i) if txt_adj is not None else 0

            # # 3. 异构信息均值融合 (防特征过载)
            # num_relations = 1 + (1 if img_adj is not None else 0) + (1 if txt_adj is not None else 0)
            # next_i = (next_i_cf + next_i_v + next_i_t) / num_relations

            # ========== 【核心修改】：非对称残差融合 ==========
            # 不要除以 3！让 CF 信号占主导 (基底)，多模态信号作为辅助残差 (10% ~ 20% 权重)
            beta = 0.2
            next_i = next_i_cf + beta * next_i_v + beta * next_i_t
            # ====================================================

            u_embs.append(next_u)
            i_embs.append(next_i)

            curr_u = next_u
            curr_i = next_i

        # 4. 层间均值聚合 (LightGCN 风格，防止过平滑)
        final_u = torch.stack(u_embs, dim=1).mean(dim=1)
        final_i = torch.stack(i_embs, dim=1).mean(dim=1)

        return final_u, final_i


class HeteroMMHAC(nn.Module):
    def __init__(self, data_config, args):    #    def __init__(self, data_config, args, pyg_data)
        super(HeteroMMHAC, self).__init__()
        self.args = args
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size

        # 1. 基础 Embeddings
        self.embedding_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embedding_item = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

        # 2. HGT 与 模态投影


        # ========== 替换原有的 self.metadata 和 self.hetero_conv ==========
        # 接收主程序传来的带有权重的多模态图
        self.img_adj = data_config.get('img_adj', None)
        self.txt_adj = data_config.get('txt_adj', None)

        # 引入新的轻量级异构图卷积
        self.hetero_conv = LightHeteroConv(
            num_layers=getattr(args, 'UI_layers', 3),
            n_users=self.n_users,
            n_items=self.n_items
        )


        # 必须给原始高维特征加 L2 归一化，防止数值爆炸！
        raw_v = torch.tensor(data_config['v_feat'], dtype=torch.float).cuda()
        raw_t = torch.tensor(data_config['t_feat'], dtype=torch.float).cuda()
        self.v_feat = F.normalize(raw_v, p=2, dim=1)
        self.t_feat = F.normalize(raw_t, p=2, dim=1)

        self.v_dense = nn.Linear(self.v_feat.shape[1], self.emb_dim)
        self.t_dense = nn.Linear(self.t_feat.shape[1], self.emb_dim)


        # [正确修正] 使用单头注意力机制，只需传入维度
        self.mod_att = MultiHeadModalityAttention(self.emb_dim)



        # 3. 聚类中心
        self.n_clusters = args.n_clusters
        self.user_centroids = nn.Parameter(torch.zeros(self.n_clusters, self.emb_dim), requires_grad=False)
        self.item_centroids = nn.Parameter(torch.zeros(self.n_clusters, self.emb_dim), requires_grad=False)

        # 4. 图结构 & 超参数
        self.Graph = data_config['ui_graph']
        if torch.cuda.is_available():
            self.Graph = self.Graph.cuda()

        # [修改] 将 tau 变为可学习参数
        self.tau = args.temperature
        # self.tau = nn.Parameter(torch.tensor(args.temperature), requires_grad=True)
        self.eps = args.eps  # 噪声系数



    # ... (perform_clustering 和 update_clusters 保持不变，省略以节省空间) ...
    def perform_clustering(self, embeddings, n_clusters):
        with torch.no_grad():
            # 新增：先做 L2 归一化
            embeddings = F.normalize(embeddings, p=2, dim=1)

            emb_np = embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(emb_np)
            return torch.tensor(kmeans.cluster_centers_, device=embeddings.device)


    def update_clusters(self, first_time=False):
        # [Fix] 获取当前模型所在的设备 (动态获取，最稳健)
        device = self.embedding_user.weight.device

        if first_time:
            # ============================================
            # 模式 A: KMeans 初始化 (只跑一次)
            # ============================================
            print("Initializing Clusters with KMeans (Cold Start)...")
            with torch.no_grad():
                # User Clustering
                user_emb = self.embedding_user.weight.detach().cpu().numpy()
                kmeans_u = KMeans(n_clusters=self.n_clusters, n_init=10).fit(user_emb)
                # [Fix] 使用 device 变量
                self.user_centroids.data = torch.tensor(kmeans_u.cluster_centers_).to(device)

                # Item Clustering
                item_emb = self.embedding_item.weight.detach().cpu().numpy()
                kmeans_i = KMeans(n_clusters=self.n_clusters, n_init=10).fit(item_emb)
                # [Fix] 使用 device 变量
                self.item_centroids.data = torch.tensor(kmeans_i.cluster_centers_).to(device)
        else:
            # ============================================
            # 模式 B: 动量更新 (Momentum Update)
            # ============================================
            alpha = 0.05  # 动量系数

            with torch.no_grad():
                # --- 更新 User 质心 ---
                user_emb = self.embedding_user.weight.detach()
                user_norm = F.normalize(user_emb, p=2, dim=1)
                cen_norm = F.normalize(self.user_centroids, p=2, dim=1)

                batch_size = 4096
                user_ids = []
                for i in range(0, self.n_users, batch_size):
                    batch = user_norm[i:i + batch_size]
                    sim = torch.mm(batch, cen_norm.t())
                    user_ids.append(sim.argmax(dim=1))
                user_ids = torch.cat(user_ids)

                new_centroids = torch.zeros_like(self.user_centroids)
                # [Fix] 使用 device 变量
                counts = torch.zeros(self.n_clusters, 1).to(device)

                new_centroids.scatter_add_(0, user_ids.view(-1, 1).expand(-1, self.emb_dim), user_emb)
                counts.scatter_add_(0, user_ids.view(-1, 1), torch.ones_like(user_ids.view(-1, 1).float()))

                counts = torch.clamp(counts, min=1.0)
                target_centroids = new_centroids / counts

                self.user_centroids.data = (1 - alpha) * self.user_centroids.data + alpha * target_centroids

                # --- 更新 Item 质心 ---
                item_emb = self.embedding_item.weight.detach()
                item_norm = F.normalize(item_emb, p=2, dim=1)
                cen_norm_i = F.normalize(self.item_centroids, p=2, dim=1)

                item_ids = []
                for i in range(0, self.n_items, batch_size):
                    batch = item_norm[i:i + batch_size]
                    sim = torch.mm(batch, cen_norm_i.t())
                    item_ids.append(sim.argmax(dim=1))
                item_ids = torch.cat(item_ids)

                new_centroids_i = torch.zeros_like(self.item_centroids)
                # [Fix] 使用 device 变量
                counts_i = torch.zeros(self.n_clusters, 1).to(device)

                new_centroids_i.scatter_add_(0, item_ids.view(-1, 1).expand(-1, self.emb_dim), item_emb)
                counts_i.scatter_add_(0, item_ids.view(-1, 1), torch.ones_like(item_ids.view(-1, 1).float()))

                counts_i = torch.clamp(counts_i, min=1.0)
                target_centroids_i = new_centroids_i / counts_i

                self.item_centroids.data = (1 - alpha) * self.item_centroids.data + alpha * target_centroids_i

                print(f"Updated Clusters with Momentum (alpha={alpha})")

    def forward(self, pyg_data, perturb=False):
        # Step 1: 输入准备
        x_u = self.embedding_user.weight
        x_i = self.embedding_item.weight  # [重要] LightGCN 输入纯 ID，不融合模态！


        # 模态特征投影

        v_emb = self.v_dense(self.v_feat)
        t_emb = self.t_dense(self.t_feat)

        # ========== 【新增：特征 Dropout】 ==========
        # 只有在模型处于 train() 模式时，dropout 才会生效
        v_emb = F.dropout(v_emb, p=getattr(self.args, 'dropout', 0.4), training=self.training)
        t_emb = F.dropout(t_emb, p=getattr(self.args, 'dropout', 0.4), training=self.training)
        # ===========================================

        # [新逻辑] 使用 ModalityAttention 进行自适应融合
        # 1. 将模态特征放入列表
        modal_list = [v_emb, t_emb]


        # 1. 解包元组，防止 HGT 崩溃
        fused_modal, _ = self.mod_att([v_emb, t_emb])

        # 2. 致命修正：将物品 ID 融合进模态特征中，保留协同区分度！
        item_semantic_feat = fused_modal + self.embedding_item.weight

        if perturb:
            noise_u = (torch.rand_like(x_u) * 2 - 1) * self.eps
            noise_i = (torch.rand_like(x_i) * 2 - 1) * self.eps
            x_u = x_u + noise_u
            x_i = x_i + noise_i

        ego_embeddings = torch.cat((x_u, x_i), dim=0)
        all_embeddings = [ego_embeddings]

        n_layers = getattr(self.args, 'UI_layers', 3)

        for k in range(n_layers):
            ego_embeddings = torch.sparse.mm(self.Graph, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_cf, i_g_cf = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)


        # ========== 【修改多模态图的传播】 ==========
        # 调用新的轻量级异构卷积 (传入带有权重的稀疏图)
        hgt_u, hgt_i = self.hetero_conv(
            self.embedding_user.weight,
            item_semantic_feat,
            self.Graph,  # ui_graph
            self.img_adj,  # visual_graph
            self.txt_adj  # text_graph
        )

        # 4. 固定系数融合
        alpha_semantic = 0.5
        final_u = u_g_cf + alpha_semantic * F.normalize(hgt_u, p=2, dim=1)
        final_i = i_g_cf + alpha_semantic * F.normalize(hgt_i, p=2, dim=1)

        return final_u, final_i, u_g_cf, i_g_cf, item_semantic_feat



    def calc_loss(self, users, pos_items, neg_items, pyg_data, epoch=0, current_ssl_reg=None):
        # 1. 获取动态权重
        if current_ssl_reg is None:
            current_ssl_reg = self.args.ssl_reg

        # 2. 前向传播
        final_u, final_i, u_cf, i_cf, h_modal = self.forward(pyg_data, perturb=False)

        # 3. 获取 Embedding
        batch_u = final_u[users]  # [B, D]
        batch_pos = final_i[pos_items]  # [B, D]
        batch_neg = final_i[neg_items]  # [B, 8, D] (混合采样) 或 [B, D] (旧代码)

        # 4. BPR Loss 计算 (核心修改部分)
        # -------------------------------------------------------
        # 计算正样本得分 [B, 1]
        pos_scores = (batch_u * batch_pos).sum(dim=1, keepdim=True)

        # 计算负样本得分 (兼容单负样本和多负样本)
        if batch_neg.dim() == 3:
            # 情况 A: 多负样本 [B, K, D]
            # 需要将 batch_u 变为 [B, 1, D] 以便广播
            neg_scores = (batch_u.unsqueeze(1) * batch_neg).sum(dim=2)  # 结果 [B, K]
        else:
            # 情况 B: 单负样本 [B, D]
            neg_scores = (batch_u * batch_neg).sum(dim=1, keepdim=True)  # 结果 [B, 1]

        # 计算 BPR Loss
        # pos_scores [B, 1] 会自动广播减去 neg_scores [B, K]
        # 结果形状 [B, K]，取平均得到标量
        loss_bpr = -torch.mean(torch.nn.LogSigmoid()(pos_scores - neg_scores))
        # -------------------------------------------------------


        # 5. SSL Loss (保持不变)
        final_u_aug, final_i_aug, _, _, _ = self.forward(pyg_data, perturb=True)
        users_unique = torch.unique(users)
        items_unique = torch.unique(pos_items)
        loss_cl_u = self.cal_cl_loss([final_u[users_unique], final_u_aug[users_unique]])
        loss_cl_i = self.cal_cl_loss([final_i[items_unique], final_i_aug[items_unique]])

        # Cross-View Loss
        loss_cv = self.cal_cl_loss([i_cf[items_unique], h_modal[items_unique].detach()])

        loss_ssl = current_ssl_reg * (loss_cl_u + loss_cl_i) + (0.1 * current_ssl_reg) * loss_cv


        # 6. Proto Loss (保持不变)
        loss_proto = torch.tensor(0.0).to(final_u.device)
        if self.user_centroids is not None:
            loss_proto = self.args.proto_reg * (self.proto_loss(batch_u, self.user_centroids) +
                                                self.proto_loss(batch_pos, self.item_centroids))

        # 7. L2 Regularization (保持不变)
        loss_reg = self.args.regs * (1 / 2) * (self.embedding_user.weight[users_unique].norm(2).pow(2) +
                                               self.embedding_item.weight[items_unique].norm(2).pow(2)) / len(
            users_unique)

        return loss_bpr + loss_ssl + loss_proto + loss_reg, loss_bpr, loss_ssl, loss_proto


    def cal_cl_loss(self, views):
        z1, z2 = views
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # 限制 tau 的范围，防止数值不稳定 (例如限制在 0.1 到 1.0 之间)
        # 这是一个训练技巧，防止 tau 变成负数或过大
        # current_tau = torch.clamp(self.tau, min=0.1, max=1.0)

        pos_score = (z1 * z2).sum(dim=1)
        # pos_score = torch.exp(pos_score / current_tau)
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.matmul(z1, z2.t())
        ttl_score = torch.sum(torch.exp(ttl_score / self.tau), dim=1)

        return -torch.log(pos_score / ttl_score + 1e-8).mean()


    def proto_loss(self, embeddings, centroids):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        centroids = F.normalize(centroids, p=2, dim=1)

        # 计算相似度矩阵
        sim = torch.mm(embeddings, centroids.t()) / getattr(self.args, 'tau_clustering', 0.5)

        # 截断梯度获取伪标签 (找寻最近的簇中心)
        pseudo_labels = torch.argmax(sim.detach(), dim=1)

        # 使用标准交叉熵计算损失，防止特征空间急剧坍缩
        return F.cross_entropy(sim, pseudo_labels)

class GatedFusionLayer(nn.Module):
    def __init__(self, dim):
        super(GatedFusionLayer, self).__init__()
        self.W = nn.Linear(dim, dim)
        self.d = nn.Linear(2 * dim, 1)  # 对应论文中的 vector d
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h_direct, h_common):
        # 投影
        h_direct_proj = self.W(h_direct)
        h_common_proj = self.W(h_common)

        # 计算 e' 和 e'' (Eq. 9, 10)
        # 注意：HAS-HGNN 原文是拼接后乘向量 d^T
        cat_1 = torch.cat((h_direct_proj, h_common_proj), dim=1)
        cat_2 = torch.cat((h_common_proj, h_direct_proj), dim=1)

        e_prime = self.leakyrelu(self.d(cat_1))
        e_double_prime = self.leakyrelu(self.d(cat_2))

        # 计算 softmax 权重 alpha (Eq. 11, 12)
        # 拼接以便在维度1上做softmax
        scores = torch.cat((e_prime, e_double_prime), dim=1)
        alphas = F.softmax(scores, dim=1)

        alpha_prime = alphas[:, 0].unsqueeze(1)
        alpha_double_prime = alphas[:, 1].unsqueeze(1)

        # 加权融合 (Eq. 13)
        h_final = alpha_prime * h_direct + alpha_double_prime * h_common
        return h_final