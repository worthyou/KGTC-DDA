import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics

from data import DataLoader
from model import DrugDiseaseModel
from evaluate import evaluate
from utils import *

def train(args, dataset, graph_data, cv):
    args.fdim_drug = dataset.drug_sim_features.shape[1]
    args.fdim_disease = dataset.disease_sim_features.shape[1]

    drug_graph = graph_data['drug_graph'].to(args.device)
    dis_graph = graph_data['disease_graph'].to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)

    args.rating_vals = dataset.possible_rel_values
    model = DrugDiseaseModel(args=args).to(args.device)

    pos_weight = th.tensor([dataset.num_neg_samples / dataset.num_pos_samples]).to(args.device)
    rel_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr)

    test_loss_logger = MetricLogger(['iter', 'loss', 'auroc', 'aupr'],
                                    ['%d', '%.4f', '%.4f', '%.4f'],
                                    os.path.join(args.save_dir, f'test_metric{args.save_id}.csv'))

    train_gt_ratings = graph_data['train'][2].to(args.device)
    train_dec_graph = graph_data['train'][1].int().to(args.device)

    best_auroc, best_aupr = 0, 0
    best_y_true, best_y_score = None, None

    for iter_idx in range(1, args.train_max_iter):
        model.train()
        pred_ratings = model(train_dec_graph, drug_graph, drug_sim_feat, dis_graph, dis_sim_feat)
        loss = rel_loss(pred_ratings.squeeze(-1), train_gt_ratings)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        auroc, aupr, y_true, y_score = evaluate(args, model, graph_data,
                                                drug_graph, drug_sim_feat,
                                                dis_graph, dis_sim_feat)
        test_loss_logger.log(iter=iter_idx, loss=loss.item(), auroc=auroc, aupr=aupr)
        if iter_idx % args.train_valid_interval == 0:
            print(f"Iter={iter_idx}, loss={loss.item():.4f}, AUROC={auroc:.4f}, AUPR={aupr:.4f}")
        if auroc > best_auroc:
            best_auroc, best_aupr = auroc, aupr
            best_y_true, best_y_score = y_true, y_score

    test_loss_logger.close()
    
    # 释放CUDA内存
    th.cuda.empty_cache()
    
    return best_auroc, best_aupr, best_y_true, best_y_score

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Drug-Disease Prediction Model")
    parser.add_argument('--seed', default=125, type=int)
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--train_max_iter', type=int, default=4000)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_valid_interval', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--fdim_drug', type=int, default=128, help='Feature dimension of drug')
    parser.add_argument('--fdim_disease', type=int, default=128, help='Feature dimension of disease')
    parser.add_argument('--nhid1', type=int, default=500)
    parser.add_argument('--nhid2', type=int, default=75)
    parser.add_argument('--nhid3', type=int, default=75, help='Hidden dimension for third GCN layer')
    parser.add_argument('--train_lr', type=float, default=0.001)
    parser.add_argument('--data_name', default='Cdataset', type=str)
    parser.add_argument('--num_neighbor', type=int, default=4)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--num_heads', type=int, default=3, help='Number of attention heads')
    parser.add_argument('--trans_num_heads', type=int, default=4, help='Number of heads in Graph Transformer')
    parser.add_argument('--use_gin', action='store_true', default=True)
    parser.add_argument('--use_transformer', action='store_true', default=True)
    args = parser.parse_args()
    
    args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
        
    aucs, auprs = [], []
    fprs, tprs, precisions, recalls = [], [], [], []
    
    for times in range(1, 2):
        print(f"+++++ times {times} +++++")
        args.save_dir = os.path.join("neighbor_num1", f"{args.data_name}_{times}time")
        os.makedirs(args.save_dir, exist_ok=True)
        dataset = DataLoader(args.data_name, args.device, k=args.num_neighbor)
        print("Dataset loaded")

        for cv in range(10):
            args.save_id = cv + 1
            print(f"==== Fold {cv + 1} ====")
            graph_data = dataset.data_cv[cv]
            auroc, auprc, y_true, y_score = train(args, dataset, graph_data, cv)
            aucs.append(auroc)
            auprs.append(auprc)

            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
            precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)

            fprs.append(fpr); tprs.append(tpr)
            precisions.append(precision); recalls.append(recall)

        print(f"AUROC Mean: {np.mean(aucs):.4f}, SE: {np.std(aucs)/np.sqrt(len(aucs)):.4f}")
        print(f"AUPRC Mean: {np.mean(auprs):.4f}, SE: {np.std(auprs)/np.sqrt(len(auprs)):.4f}")

    else:
        print("No valid AUROC and AUPRC values were calculated.")