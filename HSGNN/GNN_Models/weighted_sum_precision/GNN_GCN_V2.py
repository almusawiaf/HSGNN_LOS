import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class WeightedSumGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_matrices):
        super(WeightedSumGCN, self).__init__()
        
        self.weights = nn.Parameter(torch.ones(num_matrices) / num_matrices)
        
        self.gcn = pyg_nn.GCNConv(in_channels, out_channels)
    
    def forward(self, data, A_list):
        # Ensure weights sum to 1 using softmax
        w = F.softmax(self.weights, dim=0)
        
        # Calculate the weighted sum of adjacency matrices
        A_meta = sum(w_k * A_k for w_k, A_k in zip(w, A_list))
        
        # Use the resulting A_meta in the GCN layer
        x, edge_index = data.x, data.edge_index
        edge_weight = A_meta[edge_index[0], edge_index[1]]
        
        # Perform GCN forward pass
        x = self.gcn(x, edge_index, edge_weight)
        
        return x


