import torch
from data import get_data_cached

def top_k_edges(userEmbeds, movieEmbeds, k):
    dot_prod = userEmbeds @ movieEmbeds.T
    _, topK_indices = dot_prod.flatten().topk(k=k)
    numCols = movieEmbeds.shape[0]
    rows = torch.div(topK_indices, numCols, rounding_mode='floor')
    cols = topK_indices % numCols
    # sanity check with dot_prod[rows, cols]
    return rows, cols

def true_pos(edges_pred, gt_edges):
    cnt = 0
    for edge in edges_pred:
        if edge in gt_edges:
            cnt += 1
    return cnt

def false_pos(edges_pred, gt_edges):
    cnt = 0
    for edge in edges_pred:
        if edge not in gt_edges:
            cnt += 1
    return cnt

def false_neg(edges_pred, gt_edges):
    gt_edges_cp = gt_edges.copy()
    for edge in edges_pred:
        if edge in gt_edges_cp:
            gt_edges_cp.remove(edge)
    return len(gt_edges_cp)

def metric_wrap(userEmbeds, movieEmbeds, k, gt_edges, metric):
    rows, cols = top_k_edges(userEmbeds, movieEmbeds, k)
    # currently seems to be faster if we did set lookup, unclear if there's a pytorch vectorized way
    edges_pred = [tuple(edge) for edge in zip(rows.tolist(), cols.tolist())]
    if metric == "recall":
        return recall(edges_pred, gt_edges)
    elif metric == "precision":
        return precision(edges_pred, gt_edges)
    elif metric == "precision_recall":
        return precision_recall(edges_pred, gt_edges)
    return NotImplementedError

def precision_recall(edges, gt_edges):
    tp_count = true_pos(edges, gt_edges)
    fn_count = false_neg(edges, gt_edges)
    fp_count = false_pos(edges, gt_edges)
    # print(tp_count, fn_count, fp_count)
    try:
        prec = tp_count / (tp_count + fp_count)
    except ZeroDivisionError:
        prec = None
    try:
        rec = tp_count / (tp_count + fn_count)
    except ZeroDivisionError:
        rec = None

    return prec, rec 

def recall(edges, gt_edges):
    tp_count = true_pos(edges, gt_edges)
    fn_count = false_neg(edges, gt_edges)
    return tp_count / (tp_count + fn_count)
    
def precision(edges, gt_edges):
    tp_count = true_pos(edges, gt_edges)
    fp_count = false_pos(edges, gt_edges)
    return tp_count / (tp_count + fp_count)

import pdb
def main():
    nU = 610
    nM = 9724
    dim = 128
    print(f"{nU * nM} possible edges")
    k = 1000
    embeds = torch.rand((nU + nM, dim))
    userEmbeds = embeds[:nU]
    movieEmbeds = embeds[nU:]
    edges = top_k_edges(userEmbeds, movieEmbeds, k)
    
    data, num_users, num_movies, edge_set, mp_mask, sup_mask, val_mask, test_mask = get_data_cached()
    precision, recall = metric_wrap(userEmbeds, movieEmbeds, k, edge_set, "precision_recall")
    print(precision, recall)
        
if __name__ == "__main__":
    main()