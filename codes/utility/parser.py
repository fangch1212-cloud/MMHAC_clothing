import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='Clothing',
                        help='Choose a dataset from {Tiktok,Sports,Clothing}')
    parser.add_argument('--verbose', type=int, default=5,#5
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=400,#1000,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--regs', type=float, default=0.01,#0.01
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001,     #0.001-》0.0005
                        help='Learning rate.')
    parser.add_argument('--train_dir', default='train')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--weight_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--core', type=int, default=5,
                        help='5-core for warm-start; 0-core for cold start')
    parser.add_argument('--topk', type=int, default=10,
                        help='K value of k-NN sparsification')
    parser.add_argument('--cf_model', nargs='?', default='LightGCN',
                        help='Downstream Collaborative Filtering model {MF, NGCF, LightGCN}')
    parser.add_argument('--early_stopping_patience', type=int, default=40,
                        help='')

    parser.add_argument('--sparse', type=int, default=0, help='Sparse or dense adjacency matrix')
    parser.add_argument('--debug', default="True")

    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')

    parser.add_argument('--Ks', nargs='?', default='[10,20]',
                        help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


    parser.add_argument('--UI_layers', type=int, default=3,
                        help='UI GNN layers')
    parser.add_argument('--User_layers', type=int, default=2,
                        help='UI GNN layers')
    parser.add_argument('--Item_layers', type=int, default=2,
                        help='UI GNN layers')
    parser.add_argument('--user_loss_ratio', type=float, default=0.1,
                        help='Control the effect of the contrastive auxiliary task')
    parser.add_argument('--item_loss_ratio', type=float, default=0.7,
                        help='Control the effect of the contrastive auxiliary task')
    parser.add_argument('--temperature', type=float, default=0.3,       #--0.3
                        help='InfoNCE temperature')

    parser.add_argument('--ablation_target', type=str, default="",
                        help='UI GNN layers')

    # 在 arguments parser 中添加
    parser.add_argument('--u2u_layers', type=int, default=0, help='HAS-HGNN user layers')
    parser.add_argument('--i2i_layers', type=int, default=2, help='MMHCL item layers')
    parser.add_argument('--ssl_reg', type=float, default=0.02, help='Contrastive loss weight')#0.015--》0.03--》0.02
    parser.add_argument('--modal_align_reg', type=float, default=0.05,
                        help='Weight of cross-modal contrastive learning loss (intra-item)')
    parser.add_argument('--eps', type=float, default=0.1, help='Noise scale for SimGCL perturbation')#0.2
    # parser.py 添加:
    parser.add_argument('--proto_reg', type=float, default=0.03, help='Prototype contrastive loss weight')# 0.003
    # [新增] MM-HAC v3.0 聚类数量参数
    parser.add_argument('--n_clusters', type=int, default=500,
                        help='Number of clusters for hybrid clustering (MM-HAC v3.0). Recommended: 200-500 for Sports/Clothing.')

    # [通用] 模型正则化参数 (防止过拟合，LightGCN/MMRec 常用)
    parser.add_argument('--dropout', type=float, default=0.2,   #0.4--》0.2
                        help='Dropout rate for embeddings or dense layers.')
    parser.add_argument('--keep_prob', type=float, default=0.6,
                        help='Keep probability for edge dropout in graph adjacency matrix (LightGCN style).')

    # [系统] 性能优化参数
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader.')

    # [MM-HAC] 聚类控制参数 (关键：控制聚类频率以平衡速度和效果)
    parser.add_argument('--cluster_interval', type=int, default=20,
                        help='Interval (epochs) for updating clusters/prototypes. 1 means update every epoch.')
    parser.add_argument('--warmup_epoch', type=int, default=80,
                        help='Number of warmup epochs before starting clustering/contrastive learning.')

    # [新增] 其他可能需要的参数
    parser.add_argument('--hash_k', type=int, default=5,
                        help='Number of hash functions if using LSH for clustering acceleration.')
    # 建议在 parser.py 中确认有这个参数，或者直接用默认值 0.1/0.2
    parser.add_argument('--tau_clustering', type=float, default=0.3,        #0.2
                        help='Temperature for clustering loss.')

    # 添加是否恢复训练的参数
    parser.add_argument('--restore', type=int, default=0, help='1: restore from best_model.pth, 0: train from scratch')


    return parser.parse_args()
