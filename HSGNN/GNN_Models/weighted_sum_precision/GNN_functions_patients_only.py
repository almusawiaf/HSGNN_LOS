import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.sparse
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix 

import torch
from torch_geometric.data import Data
from torch.optim import Adam
from torch_geometric.utils import from_scipy_sparse_matrix

from torch_geometric.nn import GCNConv

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


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


from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


def top_k_precision(y_true, y_scores, k=15):
    """
    Calculate top-k precision for multilabel classification.

    Parameters:
    y_true (numpy array): Binary matrix of true labels (shape: n_samples x n_classes).
    y_scores (numpy array): Matrix of predicted scores (shape: n_samples x n_classes).
    k (int): Number of top elements to consider for calculating precision.

    Returns:
    float: Mean top-k precision across all samples.
    """
    n_samples = y_true.shape[0]
    top_k_precisions = []

    for i in range(n_samples):
        # Get the indices of the top-k predictions
        top_k_indices = np.argsort(y_scores[i])[-k:]
        
        # Calculate precision for this sample
        precision = np.sum(y_true[i, top_k_indices]) / k
        top_k_precisions.append(precision)
    
    return np.mean(top_k_precisions)

def evaluate(model, data, mask, k):
    model.eval()
    with torch.no_grad():
        # Get model predictions
        logits = model(data)
        preds = torch.sigmoid(logits[mask])
        binary_preds = (preds > 0.5).float()
        
        # Ground truth
        true_labels = data.y[mask]
        
        # Convert to CPU and numpy for sklearn metrics
        true_labels_np = true_labels.cpu().numpy()
        binary_preds_np = binary_preds.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels_np, binary_preds_np)
        
        # Calculate Hamming loss
        hamming = hamming_loss(true_labels_np, binary_preds_np)
        
        # Calculate precision, recall, F1 score for micro and macro averaging
        precision_micro = precision_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        
        precision_macro = precision_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        
        # Calculate AUC only if there are both positive and negative samples for each label
        try:
            auc = roc_auc_score(true_labels_np, preds_np, average='macro', multi_class='ovr') if len(np.unique(true_labels_np)) > 1 else 0
        except ValueError:
            auc = 0
        
        return {
            'accuracy': accuracy,
            'micro_precision': precision_micro,
            'macro_precision': precision_macro,
            'recall': recall_macro,
            'f1_score': f1_macro,
            'auc': auc,
            # Include top-k precision
            'top_k_1': top_k_precision(true_labels_np, preds_np, k=10),
            'top_k_pred': top_k_precision(true_labels_np, preds, k=4),
            # 'hamming_loss': hamming,
            # 'recall_micro': recall_micro,
            # 'f1_micro': f1_micro,
        }

def evaluate2(model, data, mask):
    model.eval()
    with torch.no_grad():
        # Get model predictions
        logits = model(data)
        preds = torch.sigmoid(logits[mask])
        binary_preds = (preds > 0.5).float()
        
        # Ground truth
        true_labels = data.y[mask]
        
        # Convert to CPU and numpy for sklearn metrics
        true_labels_np = true_labels.cpu().numpy()
        binary_preds_np = binary_preds.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels_np, binary_preds_np)
        
        # Calculate Hamming loss
        hamming = hamming_loss(true_labels_np, binary_preds_np)
        
        # Calculate precision, recall, F1 score for micro and macro averaging
        precision_micro = precision_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        
        precision_macro = precision_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        
        # Calculate AUC only if there are both positive and negative samples for each label
        try:
            auc = roc_auc_score(true_labels_np, preds_np, average='macro', multi_class='ovr') if len(np.unique(true_labels_np)) > 1 else 0
        except ValueError:
            auc = 0
        
        return {
            'accuracy': accuracy,
            # 'hamming_loss': hamming,
            'micro_precision': precision_micro,
            # 'recall_micro': recall_micro,
            # 'f1_micro': f1_micro,
            'macro_precision': precision_macro,
            'recall': recall_macro,
            'f1_score': f1_macro,
            'auc': auc
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

def select_top_diagnoses(Y, num_diagnoses=10):
    """
    Selects the top `num_diagnoses` most frequent diagnoses from the diagnosis matrix Y.
    
    Parameters:
    Y (numpy.ndarray): A matrix of shape (n, d) where n is the number of patients
                       and d is the number of possible diagnoses (binary features).
    num_diagnoses (int): The number of top diagnoses to select based on frequency.
    
    Returns:
    numpy.ndarray: A matrix of shape (n, num_diagnoses) with only the top most frequent diagnoses.
    """
    # Step 1: Sum each diagnosis column across all patients
    diagnosis_frequencies = Y.sum(axis=0)

    # Step 2: Get the indices of the top `num_diagnoses` most frequent diagnoses
    top_indices = diagnosis_frequencies.argsort()[-num_diagnoses:][::-1]

    # Step 3: Create the new matrix with only the top `num_diagnoses` diagnoses
    top_diagnoses_matrix = Y[:, top_indices]

    return top_diagnoses_matrix


def load_data(file_path, device, with_SNF, super_class, num_D, num_Meta_Path = 5):
    
    '''Using patients nodes for verifying only...'''

    num_As = num_Meta_Path
    if with_SNF:
        num_As =+1
    # Load data, assuming the paths are correct
    # X = torch.load(f'{file_path}/X_32.pt')
    temp_Y = torch.load(f'{file_path}/Y{super_class}.pt')
    Y = select_top_diagnoses(temp_Y, num_D)
    X = torch.load(f'{file_path}/X.pt')    
    
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

    if isinstance(X, np.ndarray):
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
    
    Patients = [n for n in Nodes if n[0]=='C']
    Nodes_indeces = {p:k for k,p in enumerate(Nodes)}
    patient_indeces = [Nodes_indeces[p] for p in Patients]
    
    total_nodes = len(Nodes)

    # P_train = load_dict_from_pickle(f'{Nodes_Path}/P_train.pkl')
    # P_test  = load_dict_from_pickle(f'{Nodes_Path}/P_test.pkl')
    # P_val   = load_dict_from_pickle(f'{Nodes_Path}/P_val.pkl')
    
    # # Reading the indeces of the patients to the given variables
    
    # train_index = [Nodes.index(p) for p in P_train]
    # test_index  = [Nodes.index(p) for p in P_test]
    # val_index   = [Nodes.index(p) for p in P_val]
    
    # Split patient indices into train, val, and test
    train_index, temp_index = train_test_split(patient_indeces, test_size=0.3, random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=0.5, random_state=42)  # 30% into 20% and 10%

    # Initialize masks for all nodes
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(total_nodes, dtype=torch.bool)

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
    # X, Y, edge_index, edge_weight, patient_indeces, total_nodes = load_data(file_path, device, with_SNF, super_class, num_Meta_Path=num_Meta_Path)
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
    num_classes = Y.size(1) 

    print(f'\t- Number of meta-paths used is :{len(data.As)}')
    print('\t- Preparing the model...')
    model = GNN_Model(num_features, num_classes, hidden_channels=203, num_matrices = len(data.As)).to(device)
    
    # optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
        
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

        val_precision = evaluate(model, data, val_mask, k=top_k)
        cur_prec = val_precision['macro_precision']
        cur_acc = val_precision['accuracy']
        cur_top_k = val_precision['top_k_pred']
        cur_top_k_1 = val_precision['top_k_1']
        
        
        print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Precision: {cur_prec:.4f}, ACC: {cur_acc:.4f}, top_k: {cur_top_k:.4f}, top_k=1: {cur_top_k_1:.4f}')

        losses.append(loss.item())
        val_precisions.append(val_precision)

        # # Extract Y[train] and out[train] and save them to CSV
        # Y_train_np = data.y[data.train_mask].cpu().numpy()  # Ground truth for the training set
        # out_train_np = out[data.train_mask].detach().cpu().numpy()  # Model output for the training set

        # # Convert to DataFrame
        # Y_train_df = pd.DataFrame(Y_train_np, columns=[f'Class_{i}' for i in range(num_classes)])
        # out_train_df = pd.DataFrame(out_train_np, columns=[f'Predicted_Class_{i}' for i in range(num_classes)])

        # # Save to CSV files
        # Y_train_df.to_csv(f'{file_path}/Y_train_epoch_{epoch}.csv', index=False)
        # out_train_df.to_csv(f'{file_path}/out_train_epoch_{epoch}.csv', index=False)
        

    test_precision = evaluate(model, data, test_mask, k=top_k)
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
        pred = (out[data.test_mask] > 0).float().cpu()  # Binarize the output (assuming binary classification)
        correct = data.y[data.test_mask].cpu()

    labels = [f'Class {i}' for i in range(num_classes)]  # Replace with actual class names if available
    create_multilabel_confusion_matrix(pred, correct, labels)
    
    print('The weights of the similarity matrices are as follow:')
    print(model.get_weights().cpu().detach().numpy())
    
    return losses, val_precisions








def create_multilabel_confusion_matrix(pred, correct, labels):
    # Compute the classification report
    classification_rep = classification_report(correct, pred, target_names=labels, zero_division=0)
    print(classification_rep)

    # report_as_dictionary = classification_report(correct, pred, target_names=labels, output_dict=True, zero_division=0)
    # print(report_as_dictionary)

