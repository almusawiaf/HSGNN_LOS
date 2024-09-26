import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.sparse
import json

from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix 

import torch
from torch_geometric.data import Data
from torch.optim import Adam
from torch_geometric.utils import from_scipy_sparse_matrix

from torch_geometric.nn import GCNConv

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

class SAGE_MLC(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, num_matrices=2):
        super(SAGE_MLC, self).__init__()
        self.weighted_sum = WeightedSumLayer(num_matrices)
        # Using only one SAGEConv layer
        self.conv1 = SAGEConv(num_features, 1, aggr="sum")  # Output size changed to 1
        # self.conv1 = SAGEConv(num_features, num_classes, aggr="sum")

    def forward(self, data):
        edge_weight_meta = self.weighted_sum(data.As)        
        mask = edge_weight_meta != 0
        edge_index = data.edge_index[:, mask]
        edge_weight_meta = edge_weight_meta[mask]
        
        x = self.conv1(data.x, edge_index)
        return x  # No sigmoid, directly return output for regression


    def get_embeddings(self, data):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # No gradient computation
            embeddings = self.forward(data)
        return embeddings


def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        preds = model(data)[mask]
        true_labels = data.y[mask]
        
        preds_np = preds.cpu().numpy()
        true_labels_np = true_labels.cpu().numpy()
        
        mse = mean_squared_error(true_labels_np, preds_np)
        mae = mean_absolute_error(true_labels_np, preds_np)
        r2 = r2_score(true_labels_np, preds_np)

        return {
            'MSE': mse,
            'MAE': mae,
            'R²': r2
        }

def reading_pickle(n):
    # print(f'Reading {n}')
    with open(f'{n}', 'rb') as f:
        data = pd.read_pickle(f)
    # print('\tDone reading...')
    return data

    
def check_for_nans(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
        print(tensor)
        return True
    return False

def load_data(file_path, device, with_SNF, super_class, num_D, num_Meta_Path = 5):
    
    '''Using patients nodes for verifying only...'''

    num_As = num_Meta_Path
    if with_SNF:
        num_As =+1
    # Load data, assuming the paths are correct
    Y = torch.load(f'{file_path}/Y.pt')
    X = torch.load(f'{file_path}/X.pt')    

    if isinstance(X, scipy.sparse.csr_matrix) or isinstance(X, scipy.sparse.csr_array):
        X = torch.tensor(X.toarray(), dtype=torch.float).to(device)  # Convert sparse matrix to dense and then to tensor
    else:
        X = X.to(device)

    # reading patient information...
    Nodes = load_dict_from_pickle(f'{file_path}/Nodes.pkl')
    patient_indices = [i for i, node in enumerate(Nodes) if node[0]=='C']  # Identify patient nodes
    num_patients = len(patient_indices)
    total_nodes = len(Nodes)
    del Nodes
        
    e = reading_pickle(f'{file_path}/edges/edge_list.pkl')
    sources, targets = zip(*e)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_index = edge_index.to(device)

    edge_weight = [torch.tensor(reading_pickle(f'{file_path}/edges/edge_weight{i}.pkl')).to(device) for i in range(num_As)]

    # Convert SciPy sparse matrix (csr_matrix) to a dense tensor
    if isinstance(X, scipy.sparse.csr_matrix):
        X = torch.tensor(X.toarray(), dtype=torch.float).to(device)  # Convert to dense and then to tensor
    elif isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float).to(device)
    else:
        X = X.to(device)

    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float).to(device)  # Ensuring Y is also a float for BCEWithLogitsLoss
    else:
        Y = Y.to(device)

    # Normalize X
    # X = (X - X.mean(dim=0)) / X.std(dim=0)

    # Check for NaNs in data
    if check_for_nans(X, "X") or check_for_nans(Y, "Y") or any(check_for_nans(w, f"edge_weight[{i}]") for i, w in enumerate(edge_weight)):
        raise ValueError("NaN detected in input data.")

    X = torch.nan_to_num(X)
    Y = torch.nan_to_num(Y)

    return X, Y, edge_index, edge_weight, patient_indices, total_nodes

def prepare_masks(Nodes_Path):
    # Reading the Nodes, and all other splits...
    Nodes   = load_dict_from_pickle(f'{Nodes_Path}/Nodes.pkl')
    P_train = load_dict_from_pickle(f'{Nodes_Path}/V_train.pkl')
    P_test  = load_dict_from_pickle(f'{Nodes_Path}/V_test.pkl')
    P_val   = load_dict_from_pickle(f'{Nodes_Path}/V_val.pkl')
    
    total_nodes = len(Nodes)
    # Reading the indeces of the patients to the given variables
    
    train_index = [Nodes.index(p) for p in P_train]
    test_index  = [Nodes.index(p) for p in P_test]
    val_index   = [Nodes.index(p) for p in P_val]
    
    # # Split patient indices into train, val, and test
    # train_index, temp_index = train_test_split(patient_indices, test_size=test_size, random_state=42)
    # val_index, test_index = train_test_split(temp_index, test_size=0.6667, random_state=42)  # 30% into 20% and 10%

    # Initialize masks for all nodes
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    # Set mask values to True for patient nodes
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    return train_mask, val_mask, test_mask

def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()




def init_weights(m):
    """
    Initialize weights of the model layers.

    Parameters:
    m (torch.nn.Module): The module (layer) to initialize.
    """
    if isinstance(m, GCNConv):
        for param in m.parameters():
            if param.requires_grad and param.dim() >= 2:
                torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif param.requires_grad and param.dim() < 2:
                torch.nn.init.zeros_(param)
    elif isinstance(m, torch.nn.Linear):
        if m.weight is not None and m.weight.dim() >= 2:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None and m.bias.dim() >= 1:
            torch.nn.init.zeros_(m.bias)
    # Add support for other layers as needed
    elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
        if m.weight is not None and m.weight.dim() >= 2:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None and m.bias.dim() >= 1:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
        if m.weight is not None and m.weight.dim() >= 1:
            torch.nn.init.ones_(m.weight)
        if m.bias is not None and m.bias.dim() >= 1:
            torch.nn.init.zeros_(m.bias)
    # Example for layers needing Xavier initialization
    elif isinstance(m, torch.nn.Embedding):
        if m.weight is not None and m.weight.dim() >= 2:
            torch.nn.init.xavier_uniform_(m.weight)
    
def main(file_path, 
         GNN_Model, 
         num_epochs, 
         with_SNF = False, 
         lr=1e-5, 
         exp_name = 'emb_result',
         super_class = '',
         num_Meta_Path = 5,
         top_k = 15,
         num_D = 10):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'\t- Loading the data...\n{file_path}')
    X, Y, edge_index, edge_weight, _, total_nodes = load_data(file_path, device, with_SNF, super_class, num_Meta_Path=num_Meta_Path, num_D = num_D)
    
    print('\t- Generating the Data structure ...')
    data = Data(x=X, y=Y, edge_index=edge_index, As=edge_weight) 
    
    if check_for_nans(data.x, "data.x"):
        raise ValueError("NaNs detected in input features")
    if check_for_nans(data.edge_index, "data.edge_index"):
        raise ValueError("NaNs detected in edge index")
    for i, edge_weight in enumerate(data.As):
        if check_for_nans(edge_weight, f"data.As[{i}]"):
            raise ValueError(f"NaNs detected in edge weight matrix {i}")

    print('\t- Generating the train, test, and validation sets...')
    
    train_mask, val_mask, test_mask = prepare_masks(file_path)
    data.train_mask = train_mask.to(device)
    data.val_mask   = val_mask.to(device)
    data.test_mask  = test_mask.to(device)

    num_features = X.size(1)
    num_classes = 0

    print(f'\t- Number of meta-paths used is :{len(data.As)}')
    print('\t- Preparing the model...')
    model = GNN_Model(num_features, num_classes, hidden_channels=203, num_matrices = len(data.As)).to(device)
    
    # optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
        
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss() 


    losses = []
    model.apply(init_weights)  # Apply weight initialization

    print('\t- Epochs...')
    for epoch in range(1, num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data).squeeze(1)
        
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, data, val_mask)
        print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.4f}, MSE: {val_metrics["MSE"]:.4f}, MAE: {val_metrics["MAE"]:.4f}, R²: {val_metrics["R²"]:.4f}')

        

    test_precision = evaluate(model, data, test_mask)

    results_cleaned = {key: float(value) for key, value in test_precision.items()}
    print(json.dumps(results_cleaned, indent=4))

    # Extract and save embeddings
    print('\t- Extracting and saving embeddings...')
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        embeddings = model.get_embeddings(data)
    embeddings_np = embeddings.cpu().numpy()  # Convert to NumPy array
    # Save embeddings to a .npy file
    np.save(f'{file_path}/{exp_name}.npy', embeddings_np)
    
    # Predictions for the test set
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[data.test_mask].cpu()  # Binarize the output (assuming binary classification)
        correct = data.y[data.test_mask].cpu()

    return losses







