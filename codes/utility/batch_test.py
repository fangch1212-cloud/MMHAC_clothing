import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import Data
import torch
import numpy as np

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


def test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    u_batch_size = BATCH_SIZE
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    # 确定最大的K值，用于GPU上的TopK筛选
    max_K = max(Ks)

    count = 0

    # 将Item Embeddings转置，准备进行矩阵乘法
    # shape: (embedding_dim, n_items)
    ia_embeddings = ia_embeddings.t()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]

        if len(user_batch) == 0:
            continue

        # 1. 计算分数 (GPU Matrix Multiplication)
        # u_g_embeddings: (batch_size, dim)
        # rate_batch: (batch_size, n_items)
        u_g_embeddings = ua_embeddings[user_batch]
        rate_batch = torch.matmul(u_g_embeddings, ia_embeddings)

        # 2. 屏蔽训练集中的物品 (Filter out training items)
        # 将训练集物品的分数设为极小值，使其不会出现在TopK中
        for i, user_id in enumerate(user_batch):
            try:
                train_items = data_generator.train_items[user_id]
                # 注意：这里直接在GPU tensor上操作
                rate_batch[i][train_items] = -1e9
            except Exception:
                pass

        # 3. GPU Top-K (加速核心)
        # 不再将全量数据传回CPU排序，而是直接在GPU取TopK
        _, top_k_indices = torch.topk(rate_batch, max_K)

        # 将TopK索引转回CPU进行metric计算
        top_k_indices = top_k_indices.cpu().numpy()

        # 4. 计算 Metrics
        # 这一步在CPU上进行，但只处理TopK个数据，速度很快
        for i, user_id in enumerate(user_batch):
            # 获取该用户的真实测试集 (Ground Truth)
            if is_val:
                user_pos_test = data_generator.val_set.get(user_id, [])
            else:
                user_pos_test = data_generator.test_set.get(user_id, [])

            if len(user_pos_test) == 0:
                continue

            # 生成 hit list (1 if item in ground truth else 0)
            pred_items = top_k_indices[i]

            # 优化：使用set查找加速
            ground_truth_set = set(user_pos_test)
            r = []
            for item in pred_items:
                if item in ground_truth_set:
                    r.append(1)
                else:
                    r.append(0)

            # 计算各项指标
            # 注意：这里调用 metrics 库的函数，假设其接口没变
            for k_idx, K in enumerate(Ks):
                # 截取前K个结果
                r_at_k = r[:K]

                result['precision'][k_idx] += metrics.precision_at_k(r_at_k, K)
                result['recall'][k_idx] += metrics.recall_at_k(r_at_k, K, len(user_pos_test))
                result['ndcg'][k_idx] += metrics.ndcg_at_k(r_at_k, K)
                result['hit_ratio'][k_idx] += metrics.hit_at_k(r_at_k, K)

            # AUC通常需要负样本或全排序，TopK优化下计算全量AUC会很慢。
            # 这里如果不做全量排序，AUC通常设为0或仅在需要时单独计算。
            # 考虑到速度，此处暂略过AUC或保持为0，如果必须计算建议单独抽样。
            result['auc'] += 0.

        count += len(user_batch)

    # 计算平均值
    if count > 0:
        result['precision'] /= count
        result['recall'] /= count
        result['ndcg'] /= count
        result['hit_ratio'] /= count
        result['auc'] /= count

    return result


def Test(dataset, model, device, args):
    # 1. 切换模式
    model.eval()

    # 2. 获取所有用户的 User 和 Item Embedding
    # 对于 HeteroMMHAC，我们需要传入 pyg_data 全图进行推断
    # main.py 中的 Trainer 已经处理了 get_pyg_hetero_data
    # 但由于 Test 函数接口限制，通常我们在 Main 中已经跑过一次 forward 获取了 embedding，或者在这里跑

    with torch.no_grad():
        # 假设 model 是 HeteroMMHAC 实例
        # 注意：这里需要传入 data_generator 中的 PyG 数据
        # 为了方便，我们在 Trainer 中调用 test 时，应该传递 embedding 进来，或者让 model 内部缓存

        if hasattr(model, 'pyg_data'):
            # 如果你在 Trainer 里把 pyg_data 绑到了 model 上
            pyg_data_gpu = model.pyg_data
        else:
            # 重新构建或从 dataset 获取 (较慢，建议优化)
            pyg_data = dataset.get_pyg_hetero_data()
            pyg_data_gpu = pyg_data.to(device)

        # 获取全量 Embedding
        # [Fix] 使用 *rest 忽略后面所有的返回值，这样无论 forward 返回 4 个还是 5 个都能兼容
        ua_embeddings, ia_embeddings, *rest = model.forward(pyg_data_gpu)

    # 3. 准备测试用户列表
    test_users = list(dataset.test_set.keys())

    # 4. 执行测试
    # 调用你之前优化的 test_torch
    result = test_torch(ua_embeddings, ia_embeddings, test_users, is_val=False)

    return result