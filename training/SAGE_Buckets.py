import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.sparse
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch_geometric.data import Data
from torch.optim import Adam
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# 1. Transform Y for Binary Classification
def transform_y_for_binary_classification(Y):
    # If Y is a numpy array, convert it to a PyTorch tensor
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float)
    
        # Convert to binary: 0 if LoS < 7, 1 if LoS >= 7
    return (Y >= 7).float()

class WeightedSumLayer(torch.nn.Module):
    def __init__(self, num_matrices):
        super(WeightedSumLayer, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(num_matrices))  # Learnable weights

    def forward(self, adjacency_matrices):
        weighted_sum = sum(w * A for w, A in zip(self.weights, adjacency_matrices))
        return weighted_sum

class SAGE_BinaryClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=16, num_matrices=2):
        super(SAGE_BinaryClassifier, self).__init__()
        self.weighted_sum = WeightedSumLayer(num_matrices)
        self.conv1 = SAGEConv(num_features, hidden_channels, aggr="sum")
        self.conv2 = SAGEConv(hidden_channels, 1, aggr="sum")

    def forward(self, data):
        edge_weight_meta = self.weighted_sum(data.As)
        mask = edge_weight_meta != 0
        edge_index = data.edge_index[:, mask]
        edge_weight_meta = edge_weight_meta[mask]
        x = self.conv1(data.x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# class SAGE_BinaryClassifier(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels=16, num_matrices=2):
#         super(SAGE_BinaryClassifier, self).__init__()
#         self.weighted_sum = WeightedSumLayer(num_matrices)
#         self.conv1 = SAGEConv(num_features, 1, aggr="sum")  # Single output for binary classification

#     def forward(self, data):
#         edge_weight_meta = self.weighted_sum(data.As)
#         mask = edge_weight_meta != 0
#         edge_index = data.edge_index[:, mask]
#         edge_weight_meta = edge_weight_meta[mask]
#         x = self.conv1(data.x, edge_index)
#         return x  # Return logits (no sigmoid here)

#     def get_embeddings(self, data):
#         self.eval()
#         with torch.no_grad():
#             embeddings = self.forward(data)
#         return embeddings

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict

# 2. Modify Evaluation for Binary Classification
def evaluate_binary(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)[mask]
        preds = torch.sigmoid(logits).cpu().numpy()
        binary_preds = (preds > 0.5).astype(int)
        true_labels = data.y[mask].cpu().numpy()

        accuracy = accuracy_score(true_labels, binary_preds)
        precision = precision_score(true_labels, binary_preds)
        recall = recall_score(true_labels, binary_preds)
        f1 = f1_score(true_labels, binary_preds)
        auc = roc_auc_score(true_labels, preds)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

def load_data(file_path, device, with_SNF, super_class, num_D, num_Meta_Path=5):
    # Load data, assuming the paths are correct
    Y = torch.load(f'{file_path}/Y.pt')
    X = torch.load(f'{file_path}/X.pt')

    if isinstance(X, scipy.sparse.csr_matrix) or isinstance(X, scipy.sparse.csr_array):
        X = torch.tensor(X.toarray(), dtype=torch.float).to(device)  # Convert sparse matrix to dense and then to tensor
    else:
        X = X.to(device)
    
    # Convert LoS into binary classification target
    Y = transform_y_for_binary_classification(Y)  # Transform Y

    # reading patient information...
    Nodes = load_dict_from_pickle(f'{file_path}/Nodes.pkl')
    patient_indices = [i for i, node in enumerate(Nodes) if node[0] == 'C']  # Identify patient nodes
    total_nodes = len(Nodes)

    e = load_dict_from_pickle(f'{file_path}/edges/edge_list.pkl')
    sources, targets = zip(*e)

    edge_index = torch.tensor([sources, targets], dtype=torch.long).to(device)
    edge_weight = [torch.tensor(load_dict_from_pickle(f'{file_path}/edges/edge_weight{i}.pkl')).to(device) for i in range(num_Meta_Path)]

    return X, Y, edge_index, edge_weight, patient_indices, total_nodes

def prepare_masks(Nodes_Path):
    from sklearn.model_selection import train_test_split
    import torch
    
    # Reading the Nodes and all other splits...
    Nodes = load_dict_from_pickle(f'{Nodes_Path}/Nodes.pkl')
    total_nodes = len(Nodes)
    
    # Initialize masks for all nodes
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    # Create a dictionary to map node names to indices
    Nodes_indices = {p: i for i, p in enumerate(Nodes)}
    
    # Split the indices for the nodes with prefix 'V' (patient nodes)
    V_indices = [Nodes_indices[v] for v in Nodes if v[0] == 'V']  # Adjust 'V' if needed for your dataset
    train_indices, temp_indices = train_test_split(V_indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    # Set the appropriate masks
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data).squeeze(1)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def init_weights(m):
    if isinstance(m, SAGEConv):
        for param in m.parameters():
            if param.requires_grad:
                if param.dim() >= 2:  # Weights (for layers that need them)
                    torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
                else:  # Bias or 1D parameters
                    torch.nn.init.zeros_(param)

def main(file_path, 
         GNN_Model, 
         num_epochs, 
         with_SNF=False, 
         lr=1e-5, 
         exp_name='emb_result',
         super_class='', 
         num_Meta_Path=5, 
         num_D=10):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, Y, edge_index, edge_weight, _, total_nodes = load_data(file_path, device, with_SNF, super_class, num_Meta_Path=num_Meta_Path, num_D=num_D)

    data = Data(x=X, y=Y, edge_index=edge_index, As=edge_weight)

    train_mask, val_mask, test_mask = prepare_masks(file_path)
    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)

    num_features = X.size(1)
    model = GNN_Model(num_features, hidden_channels=203, num_matrices=len(data.As)).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary classification loss

    print(f"Class distribution: {torch.sum(data.y == 0)}, {torch.sum(data.y == 1)}")
    
    # Apply initialization
    model.apply(init_weights)

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data).squeeze(1)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_metrics = evaluate_binary(model, data, val_mask)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {val_metrics["accuracy"]:.4f}, Precision: {val_metrics["precision"]:.4f}, Recall: {val_metrics["recall"]:.4f}, F1: {val_metrics["f1_score"]:.4f}, AUC: {val_metrics["auc"]:.4f}')

    test_metrics = evaluate_binary(model, data, test_mask)
    print(json.dumps(test_metrics, indent=4))

