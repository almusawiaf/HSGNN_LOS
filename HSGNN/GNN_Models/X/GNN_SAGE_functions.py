import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.sparse
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from torch.optim import Adam
from torch_geometric.utils import from_scipy_sparse_matrix

import matplotlib.pyplot as plt




def reading_pickle(n):
    # print(f'Reading {n}')
    with open(f'{n}', 'rb') as f:
        data = pd.read_pickle(f)
    # print('\tDone reading...')
    return data

def plot_combined_metrics_and_loss(losses, val_metrics):
    """
    Plots the training loss and various validation metrics over epochs in a single plot with twin y-axes.

    Parameters:
    losses (list of float): Training losses over epochs.
    val_metrics (list of dict): Validation metrics over epochs. Each dict contains keys 'precision', 'recall',
                                'accuracy', 'f1_score', and 'auc'.
    """
    # Extract metrics from val_metrics
    val_precisions = [metric["precision"] for metric in val_metrics]
    val_recalls = [metric["recall"] for metric in val_metrics]
    val_accuracies = [metric["accuracy"] for metric in val_metrics]
    val_f1_scores = [metric["f1_score"] for metric in val_metrics]
    val_aucs = [metric["auc"] for metric in val_metrics]

    epochs = range(1, len(losses) + 1)

    # Create the figure and the first y-axis
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plotting the training loss on the first y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(epochs, losses, label='Training Loss', marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create the second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Metrics', color='tab:red')
    
    # Plotting each validation metric on the second y-axis
    ax2.plot(epochs, val_precisions, label='Validation Precision', marker='o', color='tab:orange')
    ax2.plot(epochs, val_recalls, label='Validation Recall', marker='o', color='tab:green')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o', color='tab:red')
    ax2.plot(epochs, val_f1_scores, label='Validation F1-Score', marker='o', color='tab:purple')
    ax2.plot(epochs, val_aucs, label='Validation AUC', marker='o', color='tab:brown')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Adding the title and legend
    fig.suptitle('Training Loss and Validation Metrics over Epochs')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True)
    fig.tight_layout()  # Adjust layout to not overlap with title
    plt.show()


    
def load_data(file_path, device):
    # Load data, assuming the paths are correct
    X = torch.load(f'{file_path}/X_32.pt')
    Y = torch.load(f'{file_path}/Y.pt')
    
    e = reading_pickle(f'{file_path}/edges/edge_list.pkl')
    sources, targets = zip(*e)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_index = edge_index.to(device)

    edge_weight = [torch.tensor(reading_pickle(f'{file_path}/edges/edge_weight{i}.pkl')) for i in range(15)]

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float).to(device)
    else:
        X = X.to(device)

    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float).to(device)  # Ensuring Y is also a float for BCEWithLogitsLoss
    else:
        Y = Y.to(device)

    X = torch.nan_to_num(X)
    Y = torch.nan_to_num(Y)

    return X, Y, edge_index, edge_weight

def prepare_masks(num_nodes, test_size):
    train_index, temp_index = train_test_split(np.arange(num_nodes), test_size=test_size, random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=0.6667, random_state=42)  # Adjusted test_size to split remaining 30% into 20% and 10%

    train_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(train_index), True)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(val_index), True)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(test_index), True)

    return train_mask, val_mask, test_mask


def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        # Get model predictions
        logits = model(data)
        preds = torch.sigmoid(logits[mask])
        binary_preds = (preds > 0.5).float()
        
        # Ground truth
        true_labels = data.y[mask]
        
        # Calculate metrics
        true_positive = ((binary_preds == 1) & (true_labels == 1)).sum().item()
        false_positive = ((binary_preds == 1) & (true_labels == 0)).sum().item()
        false_negative = ((binary_preds == 0) & (true_labels == 1)).sum().item()
        true_negative = ((binary_preds == 0) & (true_labels == 0)).sum().item()
        
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        auc = roc_auc_score(true_labels.cpu(), preds.cpu()) if len(torch.unique(true_labels)) > 1 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }



def save_results(losses, val_accs, file_path):
    if isinstance(losses[0], torch.Tensor):
        losses = [loss.item() for loss in losses]  # Converts each tensor to a scalar and ensures it's not on CUDA
    if isinstance(val_accs[0], torch.Tensor):
        val_accs = [val_acc.item() for val_acc in val_accs]  # Same conversion for validation accuracies

    df = pd.DataFrame({
        'Loss': losses,
        'Validation Accuracy': val_accs
    })

    df.to_csv(f'{file_path}.csv', index=False)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)

def check_for_nans(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
        return True
    return False

def main(file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, Y, edge_index, edge_weight = load_data(file_path, device)
    test_size = 0.4
    
    train_mask, val_mask, test_mask = prepare_masks(X.size(0), test_size)
    data = Data(x=X, y=Y, edge_index=edge_index, As=edge_weight) 
    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)

    num_features = X.size(1)
    num_classes = Y.size(1) 

    print(f'Number of meta-paths used is :{len(data.As)}')
    model = SAGE_MLC(num_features, num_classes, hidden_channels=16, num_matrices = len(data.As)).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    losses, val_precisions = [], []

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.apply(init_weights)  # Apply weight initialization

    for epoch in range(1, 250):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        if check_for_nans(out, "model output"):
            break
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        if check_for_nans(loss, "loss"):
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        val_precision = evaluate(model, data, val_mask)
        current_precision = val_precision['precision']
        current_accuracy = val_precision['accuracy']
        
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Val Precision: {current_precision:.4f}, ACC: {current_accuracy:.4f}')

        losses.append(loss.item())
        val_precisions.append(val_precision)

    test_precision = evaluate(model, data, test_mask)
    print(f'Test Precision: \n{test_precision}')
    return losses, val_precisions
    
