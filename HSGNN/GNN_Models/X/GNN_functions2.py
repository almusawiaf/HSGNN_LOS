import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix 

import torch
from torch_geometric.data import Data
from torch.optim import Adam
from torch_geometric.utils import from_scipy_sparse_matrix

from torch_geometric.nn import GCNConv




def plot_combined_metrics_and_loss(losses, val_metrics):
    epochs = range(1, len(losses) + 1)
    # val_precisions = [metrics['precision'] for metrics in val_metrics]
    val_recalls = [metrics['recall'] for metrics in val_metrics]
    val_accuracies = [metrics['accuracy'] for metrics in val_metrics]
    val_f1_scores = [metrics['f1_score'] for metrics in val_metrics]
    val_aucs = [metrics['auc'] for metrics in val_metrics]
    val_micro_precisions = [metrics['micro_precision'] for metrics in val_metrics]
    val_macro_precisions = [metrics['macro_precision'] for metrics in val_metrics]

    fig, ax1 = plt.subplots(figsize=(15, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, label='Loss', marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Metrics', color=color)  # we already handled the x-label with ax1
    # ax2.plot(epochs, val_precisions, label='Validation Precision', marker='o', color='tab:orange')
    ax2.plot(epochs, val_recalls, label='Recall', marker='o', color='tab:green')
    ax2.plot(epochs, val_accuracies, label='Accuracy', marker='o', color='tab:purple')
    ax2.plot(epochs, val_f1_scores, label='F1 Score', marker='o', color='tab:brown')
    ax2.plot(epochs, val_aucs, label='AUC', marker='o', color='tab:pink')
    ax2.plot(epochs, val_micro_precisions, label='Micro Precision', marker='o', color='tab:cyan')
    ax2.plot(epochs, val_macro_precisions, label='Macro Precision', marker='o', color='magenta')  # changed here
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding the title and legend
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle('Training Loss and Validation Metrics', fontsize=16)
    fig.subplots_adjust(top=0.88)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

    plt.show()


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
        
        # Micro and macro precision
        micro_precision = precision_score(true_labels.cpu(), binary_preds.cpu(), average='micro', zero_division=0)
        macro_precision = precision_score(true_labels.cpu(), binary_preds.cpu(), average='macro', zero_division=0)
        
        return {
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'micro_precision': micro_precision,
            'macro_precision': macro_precision
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


def load_data(file_path, device, with_SNF, super_class):
    num_As = 16
    if with_SNF:
        num_As = 17
    # Load data, assuming the paths are correct
    X = torch.load(f'{file_path}/X_32.pt')
    Y = torch.load(f'{file_path}/Y{super_class}.pt')
    
    e = reading_pickle(f'{file_path}/edges/edge_list.pkl')
    sources, targets = zip(*e)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_index = edge_index.to(device)

    edge_weight = [torch.tensor(reading_pickle(f'{file_path}/edges/edge_weight{i}.pkl')).to(device) for i in range(num_As)]

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float).to(device)
    else:
        X = X.to(device)

    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float).to(device)  # Ensuring Y is also a float for BCEWithLogitsLoss
    else:
        Y = Y.to(device)

    # Normalize X
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    # Check for NaNs in data
    if check_for_nans(X, "X") or check_for_nans(Y, "Y") or any(check_for_nans(w, f"edge_weight[{i}]") for i, w in enumerate(edge_weight)):
        raise ValueError("NaN detected in input data.")

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


def save_results(losses, val_metrics, file_path):
    """
    Saves the training loss and various validation metrics over epochs to a CSV file.

    Parameters:
    losses (list of float): Training losses over epochs.
    val_metrics (list of dict): Validation metrics over epochs. Each dict contains keys 'precision', 'recall',
                                'accuracy', 'f1_score', and 'auc'.
    file_path (str): The path where the CSV file will be saved.
    """
    if isinstance(losses[0], torch.Tensor):
        losses = [loss.item() for loss in losses]  # Converts each tensor to a scalar and ensures it's not on CUDA
    
    # Extract metrics from val_metrics
    # val_precisions = [metric["precision"] for metric in val_metrics]
    val_macro_precisions = [metric["macro_precision"] for metric in val_metrics]
    val_micro_precisions = [metric["micro_precision"] for metric in val_metrics]
    val_recalls = [metric["recall"] for metric in val_metrics]
    val_accuracies = [metric["accuracy"] for metric in val_metrics]
    val_f1_scores = [metric["f1_score"] for metric in val_metrics]
    val_aucs = [metric["auc"] for metric in val_metrics]
    
    # Create a DataFrame with all the metrics
    df = pd.DataFrame({
        'Loss': losses,
        'Validation Micro Precision': val_micro_precisions,
        'Validation Macro Precision': val_macro_precisions,
        'Validation Recall': val_recalls,
        'Validation Accuracy': val_accuracies,
        'Validation F1-Score': val_f1_scores,
        'Validation AUC': val_aucs
    })

    # Save the DataFrame to a CSV file
    df.to_csv(f'{file_path}.csv', index=False)


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
    
# def init_weights1(m):
#     if isinstance(m, GCNConv):
#         for param in m.parameters():
#             if param.requires_grad and param.dim() >= 2:
#                 torch.nn.init.kaiming_uniform_(param)
#     elif isinstance(m, torch.nn.Linear):
#         if m.weight is not None and m.weight.dim() >= 2:
#             torch.nn.init.kaiming_uniform_(m.weight)
#         if m.bias is not None and m.bias.dim() >= 1:
#             torch.nn.init.zeros_(m.bias)

# def init_weights0(m):
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.kaiming_uniform_(m.weight.data)


def main(file_path, 
         GNN_Model, 
         num_epochs, 
         with_SNF = False, 
         lr=1e-5, 
         exp_name = 'emb_result',
         super_class = ''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'\t- Loading the data...\n{file_path}')
    X, Y, edge_index, edge_weight = load_data(file_path, device, with_SNF, super_class)
    
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
    test_size = 0.4
    train_mask, val_mask, test_mask = prepare_masks(X.size(0), test_size)
    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)

    num_features = X.size(1)
    num_classes = Y.size(1) 

    print(f'\t- Number of meta-paths used is :{len(data.As)}')
    print('\t- Preparing the model...')
    model = GNN_Model(num_features, num_classes, hidden_channels=16, num_matrices = len(data.As)).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
        
    criterion = torch.nn.BCEWithLogitsLoss()

    losses, val_precisions = [], []

    model.apply(init_weights)  # Apply weight initialization

    print('\t- Epochs...')
    for epoch in range(1, num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        if check_for_nans(out, "model output"):
            print(out)
            break
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        if check_for_nans(loss, "loss"):
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        val_precision = evaluate(model, data, val_mask)
        current_precision = val_precision['macro_precision']
        current_accuracy = val_precision['accuracy']
        
        # print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Precision: {current_precision:.4f}, ACC: {current_accuracy:.4f}')

        losses.append(loss.item())
        val_precisions.append(val_precision)

    test_precision = evaluate(model, data, test_mask)
    print(f'Test Precision: \n{test_precision}')

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
        pred = (out[data.test_mask] > 0).float().cpu()  # Binarize the output (assuming binary classification)
        correct = data.y[data.test_mask].cpu()

    labels = [f'Class {i}' for i in range(num_classes)]  # Replace with actual class names if available
    create_multilabel_confusion_matrix(pred, correct, labels)

    return losses, val_precisions








def create_multilabel_confusion_matrix(pred, correct, labels):
    # Compute the classification report
    classification_rep = classification_report(correct, pred, target_names=labels, zero_division=0)
    print(classification_rep)

    report_as_dictionary = classification_report(correct, pred, target_names=labels, output_dict=True, zero_division=0)
    print(report_as_dictionary)



# def plot_combined_metrics_and_loss(losses, val_metrics):
#     """
#     Plots the training loss and various validation metrics over epochs in a single plot with twin y-axes.

#     Parameters:
#     losses (list of float): Training losses over epochs.
#     val_metrics (list of dict): Validation metrics over epochs. Each dict contains keys 'precision', 'recall',
#                                 'accuracy', 'f1_score', and 'auc'.
#     """
#     # Extract metrics from val_metrics
#     val_precisions = [metric["precision"] for metric in val_metrics]
#     val_recalls = [metric["recall"] for metric in val_metrics]
#     val_accuracies = [metric["accuracy"] for metric in val_metrics]
#     val_f1_scores = [metric["f1_score"] for metric in val_metrics]
#     val_aucs = [metric["auc"] for metric in val_metrics]

#     epochs = range(1, len(losses) + 1)

#     # Create the figure and the first y-axis
#     fig, ax1 = plt.subplots(figsize=(12, 8))

#     # Plotting the training loss on the first y-axis
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Training Loss', color='tab:blue')
#     ax1.plot(epochs, losses, label='Training Loss', marker='o', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     # Create the second y-axis
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Validation Metrics', color='tab:red')
    
#     # Plotting each validation metric on the second y-axis
#     ax2.plot(epochs, val_precisions, label='Validation Precision', marker='o', color='tab:orange')
#     ax2.plot(epochs, val_recalls, label='Validation Recall', marker='o', color='tab:green')
#     ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o', color='tab:red')
#     ax2.plot(epochs, val_f1_scores, label='Validation F1-Score', marker='o', color='tab:purple')
#     ax2.plot(epochs, val_aucs, label='Validation AUC', marker='o', color='tab:brown')
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     # Adding the title and legend
#     fig.suptitle('Training Loss and Validation Metrics over Epochs')
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')
#     ax1.grid(True)
#     fig.tight_layout()  # Adjust layout to not overlap with title
#     plt.show()





# def evaluate(model, data, mask):
#     model.eval()
#     with torch.no_grad():
#         # Get model predictions
#         logits = model(data)
#         preds = torch.sigmoid(logits[mask])
#         binary_preds = (preds > 0.5).float()
        
#         # Ground truth
#         true_labels = data.y[mask]
        
#         # Calculate metrics
#         true_positive = ((binary_preds == 1) & (true_labels == 1)).sum().item()
#         false_positive = ((binary_preds == 1) & (true_labels == 0)).sum().item()
#         false_negative = ((binary_preds == 0) & (true_labels == 1)).sum().item()
#         true_negative = ((binary_preds == 0) & (true_labels == 0)).sum().item()
        
#         precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
#         recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
#         accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
#         f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#         auc = roc_auc_score(true_labels.cpu(), preds.cpu()) if len(torch.unique(true_labels)) > 1 else 0
        
#         return {
#             'precision': precision,
#             'recall': recall,
#             'accuracy': accuracy,
#             'f1_score': f1,
#             'auc': auc
#         }


# def evaluate2(model, data, mask):
#     model.eval()
#     with torch.no_grad():
#         # Get model predictions
#         logits = model(data)
#         preds = torch.sigmoid(logits[mask])
#         binary_preds = (preds > 0.5).float()

#         # Ground truth
#         true_labels = data.y[mask]

#         # Calculate metrics
#         true_positive = ((binary_preds == 1) & (true_labels == 1)).sum().item()
#         false_positive = ((binary_preds == 1) & (true_labels == 0)).sum().item()
#         false_negative = ((binary_preds == 0) & (true_labels == 1)).sum().item()
#         true_negative = ((binary_preds == 0) & (true_labels == 0)).sum().item()

#         precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
#         recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
#         accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
#         f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#         auc = roc_auc_score(true_labels.cpu(), preds.cpu()) if len(torch.unique(true_labels)) > 1 else 0
        
#         # Micro and macro precision
#         micro_precision = precision_score(true_labels.cpu(), binary_preds.cpu(), average='micro')
#         macro_precision = precision_score(true_labels.cpu(), binary_preds.cpu(), average='macro')

#         return {
#             'precision': precision,
#             'recall': recall,
#             'accuracy': accuracy,
#             'f1_score': f1,
#             'auc': auc,
#             'micro_precision': micro_precision,
#             'macro_precision': macro_precision
#         }




# def plot_combined_metrics_and_loss2(losses, val_metrics):
#     """
#     Plots the training loss and various validation metrics over epochs in a single plot with twin y-axes.

#     Parameters:
#     losses (list of float): Training losses over epochs.
#     val_metrics (list of dict): Validation metrics over epochs. Each dict contains keys 'precision', 'recall',
#                                 'accuracy', 'f1_score', 'auc', 'micro_precision', and 'macro_precision'.
#     """
#     # Extract metrics from val_metrics
#     val_precisions = [metric["precision"] for metric in val_metrics]
#     val_recalls = [metric["recall"] for metric in val_metrics]
#     val_accuracies = [metric["accuracy"] for metric in val_metrics]
#     val_f1_scores = [metric["f1_score"] for metric in val_metrics]
#     val_aucs = [metric["auc"] for metric in val_metrics]
#     val_micro_precisions = [metric["micro_precision"] for metric in val_metrics]
#     val_macro_precisions = [metric["macro_precision"] for metric in val_metrics]

#     epochs = range(1, len(losses) + 1)

#     # Create the figure and the first y-axis
#     fig, ax1 = plt.subplots(figsize=(12, 8))

#     # Plotting the training loss on the first y-axis
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Training Loss', color='tab:blue')
#     ax1.plot(epochs, losses, label='Training Loss', marker='o', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     # Create the second y-axis
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Validation Metrics', color='tab:red')

#     # Plotting each validation metric on the second y-axis
#     ax2.plot(epochs, val_precisions, label='Validation Precision', marker='o', color='tab:orange')
#     ax2.plot(epochs, val_recalls, label='Validation Recall', marker='o', color='tab:green')
#     ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o', color='tab:red')
#     ax2.plot(epochs, val_f1_scores, label='Validation F1-Score', marker='o', color='tab:purple')
#     ax2.plot(epochs, val_aucs, label='Validation AUC', marker='o', color='tab:brown')
#     ax2.plot(epochs, val_micro_precisions, label='Validation Micro Precision', marker='o', color='tab:cyan')
#     ax2.plot(epochs, val_macro_precisions, label='Validation Macro Precision', marker='o', color='tab:magenta')
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     # Adding the title and legend
#     fig.suptitle('Training Loss and Validation Metrics over Epochs')
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')
#     ax1.grid(True)
#     fig.tight_layout()  # Adjust layout to not overlap with title
#     plt.show()



# def main1(file_path, GNN_Model, num_epochs, with_SNF = False, lr=0.01):
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     print(f'\t- Loading the data...\n{file_path}')
#     X, Y, edge_index, edge_weight = load_data(file_path, device, with_SNF)
    
#     print('\t- Generating the Data structure ...')
#     data = Data(x=X, y=Y, edge_index=edge_index, As=edge_weight) 
    
#     if check_for_nans(data.x, "data.x"):
#         raise ValueError("NaNs detected in input features")
#     if check_for_nans(data.edge_index, "data.edge_index"):
#         raise ValueError("NaNs detected in edge index")
#     for i, edge_weight in enumerate(data.As):
#         if check_for_nans(edge_weight, f"data.As[{i}]"):
#             raise ValueError(f"NaNs detected in edge weight matrix {i}")

    
#     print('\t- Generating the train, test, and validation sets...')
#     test_size = 0.4
#     train_mask, val_mask, test_mask = prepare_masks(X.size(0), test_size)
#     data.train_mask = train_mask.to(device)
#     data.val_mask = val_mask.to(device)
#     data.test_mask = test_mask.to(device)

#     num_features = X.size(1)
#     num_classes = Y.size(1) 

#     print(f'\t- Number of meta-paths used is :{len(data.As)}')
#     print('\t- Preparing the model...')
#     model = GNN_Model(num_features, num_classes, hidden_channels=16, num_matrices = len(data.As)).to(device)
    
#     # optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
#     optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
        
#     criterion = torch.nn.BCEWithLogitsLoss()

#     losses, val_precisions = [], []

#     model.apply(init_weights)  # Apply weight initialization

#     print('\t- Epochs...')
#     for epoch in range(1, num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         out = model(data)
#         if check_for_nans(out, "model output"):
#             print(out)
#             break
#         loss = criterion(out[data.train_mask], data.y[data.train_mask])
#         if check_for_nans(loss, "loss"):
#             break

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
#         optimizer.step()

#         val_precision = evaluate(model, data, val_mask)
#         current_precision = val_precision['macro_precision']
#         current_accuracy = val_precision['accuracy']
        
#         # print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Precision: {current_precision:.4f}, ACC: {current_accuracy:.4f}')

#         losses.append(loss.item())
#         val_precisions.append(val_precision)

#     test_precision = evaluate(model, data, test_mask)
#     print(f'Test Precision: \n{test_precision}')


#     # Extract and save embeddings
#     print('\t- Extracting and saving embeddings...')
#     model.eval()  # Ensure the model is in evaluation mode
#     with torch.no_grad():
#         embeddings = model.get_embeddings(data)
#     embeddings_np = embeddings.cpu().numpy()  # Convert to NumPy array
#     # Save embeddings to a .npy file
#     np.save('embeddings.npy', embeddings_np)
    

#     return losses, val_precisions