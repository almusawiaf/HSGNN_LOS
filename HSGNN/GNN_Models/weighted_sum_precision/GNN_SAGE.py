import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class WeightedSumLayer(torch.nn.Module):
    def __init__(self, num_matrices):
        super(WeightedSumLayer, self).__init__()
        # num_matrices is the number of different adjacency matrices
        self.weights = torch.nn.Parameter(torch.randn(num_matrices))  # Learnable weights

    def forward(self, adjacency_matrices):
        # Ensure adjacency_matrices is a list of tensors
        if not isinstance(adjacency_matrices, list):
            raise TypeError("adjacency_matrices must be a list of tensors.")
        # Weighted sum of the adjacency matrices
        weighted_sum = sum(w * A for w, A in zip(self.weights, adjacency_matrices))
        return weighted_sum
    
    def get_weights(self):
        return self.weights

class SAGE_MLC(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, num_matrices=2):
        super(SAGE_MLC, self).__init__()
        self.weighted_sum = WeightedSumLayer(num_matrices)
        # Using only one SAGEConv layer
        self.conv1 = SAGEConv(num_features, num_classes, aggr="sum")

    def forward(self, data, return_embeddings=False):
        edge_weight_meta = self.weighted_sum(data.As)        
        # Filter edges based on edge weights
        mask = edge_weight_meta != 0
        edge_index = data.edge_index[:, mask]
        edge_weight_meta = edge_weight_meta[mask]

        x = self.conv1(data.x, edge_index)
        if return_embeddings:
            return x  # Return embeddings at the single layer
        return x

    def get_embeddings(self, data):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # No gradient computation
            embeddings = self.forward(data, return_embeddings=True)
        return embeddings

# import torch
# from torch_geometric.nn import SAGEConv
# import torch.nn.functional as F

# class WeightedSumLayer(torch.nn.Module):
#     def __init__(self, num_matrices):
#         super(WeightedSumLayer, self).__init__()
#         # num_matrices is the number of different adjacency matrices
#         self.weights = torch.nn.Parameter(torch.randn(num_matrices))  # Learnable weights

#     def forward(self, adjacency_matrices):
#         # Ensure adjacency_matrices is a list of tensors
#         if not isinstance(adjacency_matrices, list):
#             raise TypeError("adjacency_matrices must be a list of tensors.")
#         # Weighted sum of the adjacency matrices
#         weighted_sum = sum(w * A for w, A in zip(self.weights, adjacency_matrices))
#         return weighted_sum

# class SAGE_MLC(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden_channels=16, num_matrices=2):
#         super(SAGE_MLC, self).__init__()
#         self.weighted_sum = WeightedSumLayer(num_matrices)
#         self.conv1 = SAGEConv(num_features, hidden_channels, aggr="max")
#         self.conv2 = SAGEConv(hidden_channels, num_classes, aggr="max")

#     def forward(self, data, return_embeddings=False):
#         edge_weight_meta = self.weighted_sum(data.As)        
#         # Filter edges based on edge weights
#         mask = edge_weight_meta != 0
#         edge_index = data.edge_index[:, mask]
#         edge_weight_meta = edge_weight_meta[mask]

#         x = self.conv1(data.x, edge_index)
#         x = F.relu(x)
#         if return_embeddings:
#             return x  # Return embeddings at the hidden layer
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

#     def get_embeddings(self, data):
#         self.eval()  # Ensure the model is in evaluation mode
#         with torch.no_grad():  # No gradient computation
#             embeddings = self.forward(data, return_embeddings=True)
#         return embeddings



# # import torch
# # from torch_geometric.nn import SAGEConv

# # import torch.nn.functional as F

# # class WeightedSumLayer(torch.nn.Module):
# #     def __init__(self, num_matrices):
# #         super(WeightedSumLayer, self).__init__()
# #         # num_matrices is the number of different adjacency matrices
# #         self.weights = torch.nn.Parameter(torch.randn(num_matrices))  # Learnable weights

# #     def forward(self, adjacency_matrices):
# #         # Ensure adjacency_matrices is a list of tensors
# #         if not isinstance(adjacency_matrices, list):
# #             raise TypeError("adjacency_matrices must be a list of tensors.")

# #         # Weighted sum of the adjacency matrices
# #         weighted_sum = sum(w * A for w, A in zip(self.weights, adjacency_matrices))
# #         return weighted_sum

# # class SAGE_MLC(torch.nn.Module):
# #     def __init__(self, num_features, num_classes, hidden_channels=16, num_matrices=2):
# #         super(SAGE_MLC, self).__init__()
# #         self.weighted_sum = WeightedSumLayer(num_matrices)
# #         self.conv1 = SAGEConv(num_features, hidden_channels, aggr="max")
# #         self.conv2 = SAGEConv(hidden_channels, num_classes, aggr="max")

# #     def forward(self, data):
# #         edge_weight_meta = self.weighted_sum(data.As)        
# #         # Filter edges based on edge weights
# #         mask = edge_weight_meta != 0
# #         edge_index = data.edge_index[:, mask]
# #         edge_weight_meta = edge_weight_meta[mask]

# #         x = self.conv1(data.x, edge_index)
# #         x = F.relu(x)
# #         x = F.dropout(x, p=0.5, training=self.training)
# #         x = self.conv2(x, edge_index)
# #         return x
    