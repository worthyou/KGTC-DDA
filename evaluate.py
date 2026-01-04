import torch as th
from sklearn import metrics

def evaluate(args, model, graph_data,
             drug_graph, drug_sim_feat,
             dis_graph, dis_sim_feat):
    rating_values = graph_data['test'][2]
    dec_graph = graph_data['test'][1].int().to(args.device)

    model.eval()
    with th.no_grad():
        pred_ratings = model(dec_graph,
                             drug_graph, drug_sim_feat,
                             dis_graph, dis_sim_feat)
    y_score = pred_ratings.view(-1).cpu().numpy()
    y_true = rating_values.view(-1).cpu().numpy()

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auroc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    auprc = metrics.auc(recall, precision)

    return auroc, auprc, y_true, y_score
