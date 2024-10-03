import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

def check_for_nans(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
        return True
    return False

class WeightedSumLayer(torch.nn.Module):
    def __init__(self, num_matrices):
        super(WeightedSumLayer, self).__init__()
        
        # num_matrices is the number of different adjacency matrices
        self.weights = torch.nn.Parameter(torch.randn(num_matrices))  # Learnable weights

    def forward(self, adjacency_matrices):
        # Ensure adjacency_matrices is a list of tensors
        if not isinstance(adjacency_matrices, list):
            raise TypeError("adjacency_matrices must be a list of tensors.")

        # Move weights to the same device as the adjacency matrices
        self.weights = self.weights.to(adjacency_matrices[0].device)

        # Weighted sum of the adjacency matrices
        weighted_sum = sum(w * A for w, A in zip(self.weights, adjacency_matrices))
        return weighted_sum



class GCN_MLC(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, num_matrices=2):
        super(GCN_MLC, self).__init__()
        self.weighted_sum = WeightedSumLayer(num_matrices)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        print('Number of A is ', num_matrices)

    def forward(self, data):
        edge_weight_meta = self.weighted_sum(data.As)
        
        if check_for_nans(data.x, "X"):
            raise ValueError("NaN detected in X")
        if check_for_nans(data.edge_index, "edge_index"):
            raise ValueError("NaN detected in edge_index")
        if check_for_nans(edge_weight_meta, "edge_weight_meta"):
            raise ValueError("NaN detected in edge_weight_meta")


        x = self.conv1(data.x, data.edge_index, edge_weight=edge_weight_meta)
        if check_for_nans(x, "conv1 output"):
            print(torch.isnan(x.sum(axis=1)).sum().item())
            print(x.shape)
            
            raise ValueError("NaN detected in conv1 output")

        x = F.relu(x)
        if check_for_nans(x, "ReLU output"):
            raise ValueError("NaN detected in ReLU output")

        x = F.dropout(x, p=0.5, training=self.training)
        if check_for_nans(x, "Dropout output"):
            raise ValueError("NaN detected in Dropout output")

        x = self.conv2(x, data.edge_index, edge_weight=edge_weight_meta)
        if check_for_nans(x, "conv2 output"):
            raise ValueError("NaN detected in conv2 output")

        return x
    
    
    
# class GCN_MLC(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden_channels=16, num_matrices=2):
#         super(GCN_MLC, self).__init__()
#         self.weighted_sum = WeightedSumLayer(num_matrices)
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, num_classes)
#         print('Number of A is ', num_matrices)

#     def forward(self, data):
#         edge_weight_meta = self.weighted_sum(data.As)

#         x = self.conv1(data.x, data.edge_index, edge_weight=edge_weight_meta)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, data.edge_index, edge_weight=edge_weight_meta)
#         return x
