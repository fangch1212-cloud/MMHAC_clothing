"""
Main function for HeteroMMHAC (Heterogeneous Graph Transformer + Clustering)
Modified from original MMHAC V3 main.py
"""
import math
import os
import random
import sys
import json
import pathlib
from time import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 引入 PyTorch Geometric
try:
    from torch_geometric.data import HeteroData
except ImportError:
    print("Error: torch_geometric is not installed. Please install it to run this model.")
    sys.exit(1)

from utility.load_data import Data
from utility.parser import parse_args
# [Change 1] 导入新的异质图模型类
from Models import HeteroMMHAC
from utility.batch_test import *
from utility.logging import Logger

args = parse_args()

args.model_name = 'HeteroMMHAC'
args.time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 路径设置保持不变
path_name = f"Hetero_uu_ii={args.User_layers}_{args.Item_layers}_{args.user_loss_ratio}_{args.item_loss_ratio}" \
            f"_topk={args.topk}_t={args.temperature}_regs={args.regs}_dim={args.embed_size}_{args.ablation_target}"
path = f"../{args.dataset}/{path_name}/"
record_path = f"../{args.dataset}/HeteroMM/"
pathlib.Path(f"{path}").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{record_path}").mkdir(parents=True, exist_ok=True)


class Trainer(object):
    def __init__(self, data_generator):
        self.data_generator = data_generator
        # 配置设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # [Change 2] 获取 PyG 格式的异质图数据
        # 注意：必须在 load_data.py 的 Data 类中实现 get_pyg_hetero_data()
        print("Constructing Heterogeneous Graph Data...")




        self.pyg_data = None  # 置空，我们不再需要它！

        print("Loading Symmetric Normalized KNN Graphs...")
        # [核心修复] 必须获取 norm_type="sym" 的图，这保证了 D^(-1/2) A D^(-1/2) 的 GCN 尺度
        img_adj, txt_adj, _ = self.data_generator.get_I2I_single_mat(norm_type="sym")

        if img_adj is not None: img_adj = img_adj.to(self.device)
        if txt_adj is not None: txt_adj = txt_adj.to(self.device)

        # 配置模型参数，直接将稀疏图送入 config
        self.config = {
            'n_users': data_generator.n_users,
            'n_items': data_generator.n_items,
            'v_feat': data_generator.v_feat,
            't_feat': data_generator.t_feat,
            'ui_graph': data_generator.getSparseGraph().to(self.device),
            'img_adj': img_adj,  # 传入带有权重的视觉图
            'txt_adj': txt_adj  # 传入带有权重的文本图
        }

        # 初始化异质图模型
        self.model = HeteroMMHAC(self.config, args).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # [新增] 初始化学习率调度器
        # mode='max': 因为我们要监控 Recall，Recall 越高越好
        # factor=0.5: 每次触发时，学习率变为原来的 0.5 倍
        # patience=5: 如果连续 5 次测试 (也就是 5 * 5 = 25 个 Epoch) Recall 都没有提升，就触发衰减
        # verbose=True: 打印日志，告诉你学习率变了
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )

        # [Fix] 适配 logging.py 的参数格式 (path + target)
        self.logger = Logger(
            path=f"log/{args.dataset}/",
            is_debug=True,
            target=f"{args.model_name}_{args.time_stamp}"
        )

        # 打印参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.logging(f"Total number of parameters: {total_params}")

        # [新增] 如果 args.restore == 1，则加载最佳模型
        if args.restore == 1:
            self.load_best_model()

    def test(self, epoch):
        # [Change 4] 测试前准备
        self.model.eval()

        # 临时将 pyg_data 绑定到 model 上，以便 Test 函数能访问
        # 这是一个简单的 Hack，避免修改 Test 函数签名
        self.model.pyg_data = self.pyg_data

        test_ret = Test(self.data_generator, self.model, self.device, args)
        self.logger.logging(f"Test Epoch {epoch}: {test_ret}")
        return test_ret

    # [新增] 加载模型的辅助函数
    def load_best_model(self):
        save_path = os.path.join(path, 'best_model.pth')
        if os.path.exists(save_path):
            self.logger.logging(f"Loading best model from {save_path}...")
            # 加载 state_dict
            checkpoint = torch.load(save_path, map_location=self.device)
            # 加载到模型
            self.model.load_state_dict(checkpoint)
            self.logger.logging("Model loaded successfully!")
        else:
            self.logger.logging(f"No best_model.pth found at {save_path}, training from scratch.")

    def train(self):
        stopping_step = 0
        best_recall = 0.0

        # # ==========================================
        # # [SOTA 配置] Sports数据集全局训练策略参数
        # # ==========================================
        # # 1. 采样难度调度 (Curriculum Sampling)
        #
        # phase_1_epoch = 100  # 0~200: 全随机---原100
        #
        # phase_2_epoch = 250  # 200~400: 1个半困难 (Semi-Hard)
        # # 400+:    2个半困难 (Harder)
        #
        # # 2. 动态正则化调度
        # relax_reg_epoch = 300  # 最后 100 轮放松正则化
        # # ==========================================

        # ==========================================
        # [SOTA 配置] Clothing数据集全局训练策略参数
        # ==========================================
        # 1. 采样难度调度 (Curriculum Sampling)

        phase_1_epoch = 60  # 0~200: 全随机---原100

        phase_2_epoch = 150  # 200~400: 1个半困难 (Semi-Hard)
        # 400+:    2个半困难 (Harder)

        # 2. 动态正则化调度
        relax_reg_epoch = 300  # 最后 100 轮放松正则化
        # ==========================================



        initial_regs = args.regs
        relaxed_regs = 0.001  # 冲刺阶段的正则系数

        # initial_regs = args.regs
        # relaxed_regs = initial_regs * 0.1  # 冲刺阶段将惩罚降低到原来的十分之一 (例如 0.0001)

        '''
        # ==========================================
        # [Tiktok 极简版] 去除难样本挖掘
        # ==========================================
        # 我们不再需要 phase_1_epoch, phase_2_epoch
        # 全程使用纯随机采样 (n_hard = 0)

        relax_reg_epoch = 200  # 提前放松正则化
        initial_regs = args.regs
        relaxed_regs = 0.005
        '''


        # 3. 学习率调度 (Cosine Ann ealing)
        # T_max=args.epoch: 规划整个训练周期的衰减曲线
        # eta_min=1e-6: 最终衰减到的最小值
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epoch, eta_min=1e-6
        )

        # 记录当前使用的学习率
        current_lr = args.lr

        for epoch in range(args.epoch):
            t1 = time()

            # ======================================================
            # [1. 聚类更新] (Warmup -> KMeans -> Momentum)
            # ======================================================
            if epoch >= args.warmup_epoch:
                if epoch == args.warmup_epoch:
                    self.logger.logging(f"[Epoch {epoch}] Warmup finished. Initializing clusters with KMeans...")
                    self.model.update_clusters(first_time=True)
                elif (epoch - args.warmup_epoch) % args.cluster_interval == 0:
                    # self.logger.logging(f"[Epoch {epoch}] Momentum update for clusters...")
                    self.model.update_clusters(first_time=False)

            # ======================================================
            # [2. 自适应采样策略] (3-Stage Curriculum)
            # ======================================================

            """
            ###sports数据集###
            n_hard = 0
            if epoch >= phase_2_epoch:
                n_hard = 2  # 阶段 3: 高难度
            elif epoch >= phase_1_epoch:
                n_hard = 1  # 阶段 2: 中等难度

            n_random = 8 - n_hard  # 总数保持 8
            """


            # clothing数据集
            n_hard = 0
            if epoch >= 120:
                n_hard = 1  # 阶段 2: 引入 1 个半困难样本施加压力
            n_random = 8 - n_hard



            '''
            # tiktok[关键修改] 强制关闭硬负样本
            n_hard = 0
            n_random = 8
            '''


            # ======================================================
            # [3. 动态正则化]
            # ======================================================
            current_regs = initial_regs
            if epoch >= relax_reg_epoch:
                current_regs = relaxed_regs
                # [新增] 当进入放松阶段时，我们可以打印一下提示
                if epoch == relax_reg_epoch:
                    self.logger.logging(f"*** Triggering Relaxed Regularization: regs -> {relaxed_regs} ***")

            # 始终使用低 SSL 权重以保证 BPR 主导
            current_ssl_reg = args.ssl_reg

            self.model.train()
            loss, bpr_loss, ssl_loss, proto_loss = 0., 0., 0., 0.

            n_batch = self.data_generator.n_train // args.batch_size + 1


            for idx in range(n_batch):
                users, pos_items, neg_items = self.data_generator.sample()
                batch_users = torch.LongTensor(users).to(self.device)
                batch_pos = torch.LongTensor(pos_items).to(self.device)
                batch_neg = torch.LongTensor(neg_items).to(self.device)

                # --------------------------------------------------
                # 混合负采样逻辑 (Type-Safe Fix)
                # --------------------------------------------------
                with torch.no_grad():
                    # Part A: 挖掘 Semi-Hard
                    hard_negs_list = []
                    if n_hard > 0:
                        user_emb = self.model.embedding_user(batch_users)
                        # 撒网 50 个
                        candidates = torch.randint(0, self.model.n_items, (len(batch_users), 1000)).to(self.device)
                        cand_emb = self.model.embedding_item(candidates)
                        scores = (user_emb.unsqueeze(1) * cand_emb).sum(dim=2)

                        # 取 Top-10
                        vals, indices = torch.topk(scores, k=10, dim=1)

                        for _ in range(n_hard):
                            # 从 Top-2 ~ Top-10 中随机选 (避开 Top-1)
                            rand_choice = torch.randint(1, 10, (len(batch_users), 1)).to(self.device)
                            selected_indices = indices.gather(1, rand_choice)
                            h_neg = candidates.gather(1, selected_indices)
                            hard_negs_list.append(h_neg)

                        hard_negs_tensor = torch.cat(hard_negs_list, dim=1)  # [B, n_hard]
                    else:
                        # [Fix] 显式指定 Long 类型
                        hard_negs_tensor = torch.empty((len(batch_users), 0), dtype=torch.long).to(self.device)

                    # Part B: 准备 Random
                    random_negs_list = []
                    if n_random > 0:
                        # 优先复用 data_generator 的 batch_neg
                        random_negs_list.append(batch_neg.unsqueeze(1))
                        remaining = n_random - 1
                        if remaining > 0:
                            extra_rnd = torch.randint(0, self.model.n_items, (len(batch_users), remaining)).to(
                                self.device)
                            random_negs_list.append(extra_rnd)
                        random_negs_tensor = torch.cat(random_negs_list, dim=1)
                    else:
                        # [Fix] 显式指定 Long 类型
                        random_negs_tensor = torch.empty((len(batch_users), 0), dtype=torch.long).to(self.device)

                    # Part C: 拼接
                    final_neg_items = torch.cat([hard_negs_tensor, random_negs_tensor], dim=1)



                self.optimizer.zero_grad()

                # 计算 Loss (传入所有动态参数)
                # 注意：请确保你的 Models.py 里的 calc_loss 接收 current_regs
                # 如果没改 Models.py，这里先传 args.regs，或者去改 Models.py
                # 这里假设你已经修改了 calc_loss 接口，或者我们临时用一种 Hack 方式：
                # Hack: 临时修改 args.regs (不推荐并发，但单卡没问题)
                original_regs = args.regs
                args.regs = current_regs

                # 计算 Loss
                batch_loss, batch_bpr, batch_ssl, batch_proto = self.model.calc_loss(
                    batch_users, batch_pos, final_neg_items,
                    pyg_data=None,  # 不再需要 pyg_data
                    epoch=epoch,
                    current_ssl_reg=current_ssl_reg  # 【消融验证】强制将 SSL 设为 0，观察纯结构的巅峰效果！
                )

                args.regs = original_regs  # 还原

                batch_loss.backward()
                self.optimizer.step()

                loss += batch_loss.item()
                bpr_loss += batch_bpr.item()
                ssl_loss += batch_ssl.item()
                proto_loss += batch_proto.item()


            # [Scheduler] 每个 Epoch 结束更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            if np.isnan(loss):
                self.logger.logging('ERROR: Loss is NaN')
                sys.exit()

            t2 = time()
            if epoch % args.verbose == 0:
                self.logger.logging(
                    f"Epoch {epoch}: Loss={loss / n_batch:.4f} [BPR={bpr_loss / n_batch:.4f}, SSL={ssl_loss / n_batch:.4f}, Proto={proto_loss / n_batch:.4f}, LR={current_lr:.6f}], Time={t2 - t1:.2f}s")

            if epoch % 5 == 0:
                test_ret = self.test(epoch)
                curr_recall = test_ret['recall'][1]

                if curr_recall > best_recall:
                    best_recall = curr_recall
                    stopping_step = 0
                    save_path = os.path.join(path, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.logging(f"*** New Best Recall: {best_recall:.5f} ***")
                else:
                    stopping_step += 1
                    if stopping_step >= args.early_stopping_patience:
                        self.logger.logging(f"Early stopping trigger at step: {stopping_step}")
                        break

        self.logger.logging("Training Finished.")


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seed(args.seed)

    # 初始化数据生成器
    # 注意：这里的 Data 类需要被修改以支持 PyG
    print(f"Loading Dataset: {args.dataset}")
    data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)

    # 初始化训练器
    trainer = Trainer(data_generator)

    # 开始训练
    trainer.train()