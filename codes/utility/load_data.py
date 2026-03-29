import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
import torch
import sys
from torch_geometric.data import HeteroData
from utility.parser import parse_args

args = parse_args()


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path + '/%d-core' % args.core
        self.batch_size = batch_size

        train_file = path + '/%d-core/train.json' % (args.core)
        val_file = path + '/%d-core/val.json' % (args.core)
        test_file = path + '/%d-core/test.json' % (args.core)

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0
        self.neg_pools = {}
        self.exist_users = []

        # [新增] 用于构建 PyG edge_index 的列表
        self.trainUser = []
        self.trainItem = []

        try:
            train = json.load(open(train_file))
            test = json.load(open(test_file))
            val = json.load(open(val_file))
        except Exception as e:
            print(f"Error loading JSON files: {e}")
            sys.exit(1)

        for uid, items in train.items():
            if len(items) == 0: continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

            # [新增] 填充交互列表，用于 PyG
            for i in items:
                self.trainUser.append(uid)
                self.trainItem.append(i)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0: continue
            uid = int(uid)
            for idx, i in enumerate(train_items):
                self.R[uid, i] = 1.
            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0: continue
            try:
                self.test_set[uid] = test_items
            except:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0: continue
            try:
                self.val_set[uid] = val_items
            except:
                continue

        # [新增] 预加载模态特征，供 HeteroMMHAC 模型初始化使用
        self._load_features()

    def _load_features(self):
        print("Loading raw features for model init...")
        try:
            # 尝试加载 npy 文件
            self.v_feat = np.load(f'../data/{args.dataset}/image_feat.npy')
            self.t_feat = np.load(f'../data/{args.dataset}/text_feat.npy')
            print(f"Features loaded: Visual {self.v_feat.shape}, Text {self.t_feat.shape}")
        except Exception as e:
            print(f"Warning: Could not load features (.npy). using random init. Error: {e}")
            # Fallback: 生成随机特征防止报错，或者报错退出
            self.v_feat = np.random.randn(self.n_items, args.embed_size)
            self.t_feat = np.random.randn(self.n_items, args.embed_size)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items

    # --------------------------------------- Graph Construction --------------------------------------------------

    def norm_dense(self, adj, normalization='origin'):
        if normalization == 'sym':
            rowsum = torch.sum(adj, -1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
            L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        elif normalization == 'rw':
            rowsum = torch.sum(adj, -1)
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diagflat(d_inv)
            L_norm = torch.mm(d_mat_inv, adj)
        elif normalization == 'origin':
            L_norm = adj
        return L_norm

    # ============================================================================
    # 内存安全的核心工具函数 (分块构建 KNN)
    # ============================================================================
    def build_knn_sparse_batch(self, features, topk, batch_size=2048):
        """
        分块构建 KNN 图，避免 OOM。
        """
        n_nodes = features.shape[0]
        # 归一化特征
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        rows, cols, data = [], [], []

        # 如果显存不够，可以把 features 转到 cpu: features = features.cpu()
        start = 0
        while start < n_nodes:
            end = min(start + batch_size, n_nodes)
            # 1. 计算当前 Batch 的相似度 [Batch, N]
            batch_feats = features[start:end]
            sim_batch = torch.mm(batch_feats, features.t())

            # 2. 取 Top-K
            # vals: [Batch, K], inds: [Batch, K]
            knn_val, knn_ind = torch.topk(sim_batch, topk, dim=-1)

            # 3. 构造稀疏坐标
            row_idx = torch.arange(start, end).view(-1, 1).expand(-1, topk).flatten()
            col_idx = knn_ind.flatten()

            rows.append(row_idx.cpu().numpy())
            cols.append(col_idx.cpu().numpy())
            data.append(np.ones(len(row_idx)))  # Unweighted graph

            start += batch_size
            del sim_batch, knn_val, knn_ind  # 及时释放内存

        # 4. 合并所有 Batch
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        # 5. 构建 Scipy 稀疏矩阵
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        return adj

    def norm_sparse(self, adj, norm_type='origin'):
        """
        稀疏矩阵归一化
        """
        if norm_type == 'origin':
            return adj

        adj = adj.tocsr()
        if norm_type == 'sym':
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        elif norm_type == 'rw':
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
        else:
            norm_adj = adj
        return norm_adj.tocoo()

    # ============================================================================
    # 获取各种图结构 (UI, I2I, Hypergraph)
    # ============================================================================

    def get_UI_mat(self, norm_type='sym'):
        """
        [Memory Optimized] 获取 User-Item 交互矩阵 (用于 LightGCN 骨架)
        使用 scipy.sparse 进行归一化，避免 todense() OOM
        """
        print("Loading UI_mat:(" + norm_type + ")")
        t = time()

        try:
            UI_mat = torch.load(self.path + '/UI_mat_' + norm_type + ".pth")
        except Exception:
            print(f"Generating UI_mat from scratch (Sparse Mode)...")
            R = self.R.tocsr()  # (n_users, n_items)

            # 构造大矩阵 A = [[0, R], [R.T, 0]]
            # Top part: [Zero(n_u, n_u), R]
            top = sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), R])
            # Bottom part: [R.T, Zero(n_i, n_i)]
            bottom = sp.hstack([R.T, sp.csr_matrix((self.n_items, self.n_items))])
            adj_mat = sp.vstack([top, bottom]).tocsr()

            # 添加自环 (Self-Loop): A + I
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            # 稀疏归一化
            if norm_type == 'sym':
                rowsum = np.array(adj_mat.sum(1))
                d_inv_sqrt = np.power(rowsum, -0.5).flatten()
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
                norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
                norm_adj = norm_adj.tocoo()

            elif norm_type == 'rw':
                rowsum = np.array(adj_mat.sum(1))
                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat_inv = sp.diags(d_inv)
                norm_adj = d_mat_inv.dot(adj_mat).tocoo()
            else:
                norm_adj = adj_mat.tocoo()

            # 转换为 PyTorch Sparse Tensor
            indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
            values = torch.from_numpy(norm_adj.data).float()
            shape = torch.Size(norm_adj.shape)
            UI_mat = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

            print("Saving UI_mat to cache...")
            torch.save(UI_mat, self.path + '/UI_mat_' + norm_type + ".pth")

        print("End Load UI_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_U2U_mat(self, norm_type='rw'):
        print("Loading User_mat:(" + norm_type + ")")
        t = time()
        try:
            User_mat = torch.load(self.path + '/User_mat_' + norm_type + ".pth")
        except Exception:
            R = self.R.tocsr()
            User_mat = R.dot(R.T)
            User_mat.setdiag(0)
            User_mat.eliminate_zeros()

            rowsum = np.array(User_mat.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            User_mat = d_mat_inv.dot(User_mat).tocoo()

            indices = torch.from_numpy(np.vstack((User_mat.row, User_mat.col)).astype(np.int64))
            values = torch.from_numpy(User_mat.data).float()
            shape = torch.Size(User_mat.shape)
            User_mat = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

            torch.save(User_mat, self.path + '/User_mat_' + norm_type + ".pth")
        print("End Load User_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return User_mat

    def get_I2I_single_mat(self, norm_type="sym"):
        """
        获取单模态的 KNN 稀疏图 (Image, Text)
        """
        # [修改 1] 在打印日志中加入 topk 信息
        print(f"Loading I2I media-specific mat:({norm_type})_topk:{args.topk}")
        t = time()

        # [修改 2] 将 args.topk 加入文件名，防止缓存冲突
        img_path = f"{self.path}/Image_mat_{norm_type}_topk_{args.topk}.pth"
        txt_path = f"{self.path}/Text_mat_{norm_type}_topk_{args.topk}.pth"
        aud_path = f"{self.path}/Audio_mat_{norm_type}_topk_{args.topk}.pth"

        try:
            image_adj = torch.load(img_path)
            text_adj = torch.load(txt_path)
            if args.dataset == "tiktok":
                audio_adj = torch.load(aud_path)
        except Exception:
            print("Generating I2I Single Mats from scratch...")

            if not hasattr(self, 'v_feat'): self._load_features()

            # 1. Visual
            v_feat_torch = torch.tensor(self.v_feat).float()
            image_adj_sp = self.build_knn_sparse_batch(v_feat_torch, topk=args.topk)
            image_adj_sp = self.norm_sparse(image_adj_sp, norm_type)

            i_indices = torch.from_numpy(np.vstack((image_adj_sp.row, image_adj_sp.col)).astype(np.int64))
            i_values = torch.from_numpy(image_adj_sp.data).float()
            image_adj = torch.sparse_coo_tensor(i_indices, i_values, torch.Size(image_adj_sp.shape))

            # 2. Textual
            t_feat_torch = torch.tensor(self.t_feat).float()
            text_adj_sp = self.build_knn_sparse_batch(t_feat_torch, topk=args.topk)
            text_adj_sp = self.norm_sparse(text_adj_sp, norm_type)

            t_indices = torch.from_numpy(np.vstack((text_adj_sp.row, text_adj_sp.col)).astype(np.int64))
            t_values = torch.from_numpy(text_adj_sp.data).float()
            text_adj = torch.sparse_coo_tensor(t_indices, t_values, torch.Size(text_adj_sp.shape))

            if args.dataset == "tiktok":
                try:
                    audio_feats = np.load(f'../data/{args.dataset}/audio_feat.npy')
                    a_feat_torch = torch.tensor(audio_feats).float()
                    audio_adj_sp = self.build_knn_sparse_batch(a_feat_torch, topk=args.topk)
                    audio_adj_sp = self.norm_sparse(audio_adj_sp, norm_type)

                    a_indices = torch.from_numpy(np.vstack((audio_adj_sp.row, audio_adj_sp.col)).astype(np.int64))
                    a_values = torch.from_numpy(audio_adj_sp.data).float()
                    audio_adj = torch.sparse_coo_tensor(a_indices, a_values, torch.Size(audio_adj_sp.shape))
                    torch.save(audio_adj, aud_path)
                except:
                    audio_adj = None

            # [修改 3] 按新的带 topk 的路径保存
            torch.save(image_adj, img_path)
            torch.save(text_adj, txt_path)

        print(f"End Load I2I media-specific mat:[{time() - t:.1f}s]({norm_type})")

        if args.dataset == "tiktok":
            return image_adj, text_adj, audio_adj
        else:
            return image_adj, text_adj, None

    def get_I2I_Hypergraph_mul_mat(self, norm_type="sym"):
        """
        [Memory Optimized] 获取多模态超图乘积 (H * H^T)
        """
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph_mul = torch.load(f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            # 1. 获取各个模态的 KNN 矩阵 (List of Sparse Tensors)
            image_adj, text_adj, _ = self.get_I2I_single_mat("origin")  # Use origin for hypergraph construction

            # 2. 转回 Scipy Sparse 进行拼接
            def torch_to_scipy(t_sparse):
                idx = t_sparse.coalesce().indices().cpu().numpy()
                val = t_sparse.coalesce().values().cpu().numpy()
                return sp.coo_matrix((val, (idx[0], idx[1])), shape=t_sparse.shape)

            img_sp = torch_to_scipy(image_adj)
            txt_sp = torch_to_scipy(text_adj)

            # 3. 拼接 H = [H_v, H_t]
            H = sp.hstack([img_sp, txt_sp]).tocsr()

            # 4. 计算 H * H^T
            H_mul = H.dot(H.T)

            # 5. 归一化
            H_mul = self.norm_sparse(H_mul, norm_type)

            # 6. 转回 Torch
            H_mul = H_mul.tocoo()
            indices = torch.from_numpy(np.vstack((H_mul.row, H_mul.col)).astype(np.int64))
            values = torch.from_numpy(H_mul.data).float()
            Hypergraph_mul = torch.sparse_coo_tensor(indices, values, torch.Size(H_mul.shape), dtype=torch.float32)

            torch.save(Hypergraph_mul, f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")

        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    # ============================================================================
    # PyG HeteroData 构建 (关键修复)
    # ============================================================================

    def get_pyg_hetero_data(self):
        """
        构建 PyG 的 HeteroData 对象，用于异质图神经网络 (如 HGT)
        """
        print("Constructing PyG HeteroData...")
        data = HeteroData()

        # 1. 节点数量
        data['user'].num_nodes = self.n_users
        data['item'].num_nodes = self.n_items

        # 2. 构建 User-Item 边
        if len(self.trainUser) == 0:
            print("Error: Train lists are empty. Make sure __init__ loaded json correctly.")

        ui_edge_index = torch.tensor([self.trainUser, self.trainItem], dtype=torch.long)
        data['user', 'interacts', 'item'].edge_index = ui_edge_index
        # 添加反向边 (Item -> User)
        data['item', 'rev_interacts', 'user'].edge_index = ui_edge_index.flip([0])

        # 3. 构建 Item-Item 语义边 (从 KNN 图中提取)
        # 确保加载了 KNN 图
        img_adj, txt_adj, _ = self.get_I2I_single_mat(norm_type="origin")  # 获取未归一化的邻接关系

        if img_adj is not None:
            # torch.sparse -> edge_index
            indices = img_adj.coalesce().indices()
            data['item', 'visual_sim', 'item'].edge_index = indices

        if txt_adj is not None:
            indices = txt_adj.coalesce().indices()
            data['item', 'text_sim', 'item'].edge_index = indices

        return data

    # [务必保留] 获取 SparseGraph 用于 LightGCN 兼容
    def getSparseGraph(self):
        return self.get_UI_mat()

    def build_sim(self, context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim