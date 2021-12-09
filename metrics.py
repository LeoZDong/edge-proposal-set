import torch


def top_k_edges(userEmbeds, movieEmbeds, k):
    dot_prod = userEmbeds @ movieEmbeds.T
    _, topK_indices = dot_prod.flatten().topk(k=k)
    numCols = movieEmbeds.shape[0]
    rows = topK_indices // numCols
    cols = topK_indices % numCols
    # sanity check with dot_prod[rows, cols]
    return rows, cols

def recall(embeds, k):
    pass
    
def precision(embeds, k):
    pass

import pdb
def main():
    nU = 610
    nM = 9724
    dim = 128
    k = 1000
    embeds = torch.rand((nU + nM, dim))
    userEmbeds = embeds[:nU]
    movieEmbeds = embeds[nU:]
    edges = top_k_edges(userEmbeds, movieEmbeds, k)
    pdb.set_trace()
        
if __name__ == "__main__":
    main()