import torch
from torch_geometric.data import Data
device = "cuda" if torch.cuda.is_available() else "cpu"

def knn_edges_torch(x, k):
    N = x.shape[0]
    if N == 0: return torch.empty((2,0), dtype=torch.long)
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(1e9)
    k = min(k, max(1, N-1))
    idx = dist.topk(k, largest=False).indices
    src = torch.arange(N).unsqueeze(1).repeat(1,k).reshape(-1)
    tgt = idx.reshape(-1)
    return torch.stack([src, tgt], dim=0)

def build_inter_edges_roi(N_non, region_non, region_make):
    src, tgt = [], []
    for r in region_non:
        A = region_non[r]
        B = region_make[r]
        L = min(len(A), len(B))
        for i in range(L):
            tgt.append(A[i])
            src.append(B[i] + N_non)
    if len(src)==0:
        return torch.empty((2,0), dtype=torch.long)
    return torch.tensor([src, tgt], dtype=torch.long)

def build_graph(non_xy, make_xy, non_f, make_f, rmN, rmM, k=6):
    N_non = non_xy.shape[0]
    x = torch.cat([non_f, make_f], dim=0)
    eN = knn_edges_torch(torch.tensor(non_xy, dtype=torch.float32), k)
    eM = knn_edges_torch(torch.tensor(make_xy, dtype=torch.float32), k) + N_non
    eI = build_inter_edges_roi(N_non, rmN, rmM)
    edges = []
    if eN.numel(): edges.append(eN)
    if eM.numel(): edges.append(eM)
    if eI.numel(): edges.append(eI)
    edge_index = torch.cat(edges, 1) if edges else torch.empty((2,0),dtype=torch.long)
    edge_attr = torch.zeros((edge_index.shape[1],1))
    if eI.numel(): edge_attr[-eI.shape[1]:,0]=1
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.N_non = N_non
    data.N_make = make_xy.shape[0]
    return data

def build_edge_attr(edge_index, is_makeup_mask):
    src, tgt = edge_index
    inter = ((is_makeup_mask[src]==1) & (is_makeup_mask[tgt]==0)).float()
    return inter.unsqueeze(1).to(device)