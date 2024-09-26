import numpy as np
import pandas as pd
import networkx as nx 
import pickle
import random
import torch
import scipy
import os
from scipy import sparse
from scipy.sparse import dok_matrix, lil_matrix, coo_matrix, csr_matrix


folder_path = '/home/almusawiaf/MyDocuments/PhD_Projects/Data/MIMIC_resources'


def extract3(code):
    return str(code)[:3]
def extract2(code):
    return str(code)[:2]

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict

def save_dict_to_pickle(dictionary, filename):
    import os
    print(f'Saving the dictionary to {filename}...')
    # Extract the directory path from the filename
    directory = os.path.dirname(filename)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the dictionary to the pickle file
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)
    print('Saving complete...')

# Removing <patients to delete> from HG
def remove_patients_and_linked_visits(nodes, HG):
    '''remove patients and their visits from HG'''
    print('Number of PATIENTS to remove: ', len(nodes))
    
    new_HG = deepcopy(HG)
    nodes_to_remove = []
    for node in nodes:
        for v in HG.neighbors(node):
            if v[0]=='V':
                nodes_to_remove.append(v)
        nodes_to_remove.append(node)
    print('Number of nodes to remove: ', len(nodes_to_remove))
    new_HG.remove_nodes_from(nodes_to_remove)
    return new_HG                 

def load_patients_data(num_Diseases):
    # 1. read the prescreption file...
    df_Medications = pd.read_csv(f'{folder_path}/PRESCRIPTIONS.csv')

    # 3. read the other files
    df_DiagnosisICD  = pd.read_csv(f'{folder_path}/DIAGNOSES_ICD.csv')    # Diagnosis!
    df_ProceduresICD = pd.read_csv(f'{folder_path}/PROCEDURES_ICD.csv')    # Procedures!

    # 3.1. reading all patients...
    df_admissions = pd.read_csv(f'{folder_path}/ADMISSIONS.csv')    
    patients = df_admissions['SUBJECT_ID'].unique()   

    # 4. selected the data for the selected patients only...
    print('Use the patients inside the new DF....')
    new_Diagnosis  = df_DiagnosisICD[df_DiagnosisICD['SUBJECT_ID'].isin(patients)]
    new_Procedures = df_ProceduresICD[df_ProceduresICD['SUBJECT_ID'].isin(patients)]
    new_Medication = df_Medications[df_Medications['subject_id'].isin(patients)]


    # processing the data, and modify length of the diagnosis and procedure:
    print('For the given diagnosis, extract the sub dataframe....')
    new_Diagnosis.dropna(subset=['ICD9_CODE'], inplace=True)
    new_Procedures.dropna(subset=['ICD9_CODE'], inplace=True)
    new_Medication.dropna(subset=['drug'], inplace=True)

    new_Diagnosis['ICD9_CODE']  = new_Diagnosis['ICD9_CODE'].apply(extract3)
    new_Procedures['ICD9_CODE'] = new_Procedures['ICD9_CODE'].apply(extract2)
    # ----------------------------------------------------------------------------
    
    diag_frequency = new_Diagnosis['ICD9_CODE'].value_counts().head(num_Diseases).index.tolist()
    
    print('Diagnoses frequency = ', diag_frequency)
    
    new_Diagnosis  = new_Diagnosis[new_Diagnosis['ICD9_CODE'].isin(diag_frequency)]
    
    
    # ----------------------------------------------------------------------------
    # extracting the unique sets of nodes of diff category.
    Procedures = sorted(new_Procedures['ICD9_CODE'].unique())
    Medication = sorted(new_Medication['drug'].unique())
    Diagnosis  = new_Diagnosis['ICD9_CODE'].unique()

    print('General Information:\n---------------------------')
    print(f'Number of Patients = {len(patients)}')
    print(f'Number of Diagnosis = {len(Diagnosis)}')
    print(f'Number of procedures = {len(Procedures)}')
    print(f'Number of Medication = {len(Medication)}')

    return new_Diagnosis, new_Medication, new_Procedures


def load_data(the_selected_diseases, disease_data_path, disease_file, num_Diseases):

    patients = set()
    
    for disease in the_selected_diseases:
        new_patients = load_dict_from_pickle(f'{disease_data_path}/203_Diagnoses/use_cases/{disease}/clinical_items/Patients.pkl')
        patients.update(new_patients)
    
    print(patients)
    save_dict_to_pickle(patients, f'{disease_data_path}/{num_Diseases}_Diagnoses/use_cases/{disease_file}/New_Patients.pkl')
    
    # 1. read the prescreption file...
    df_Medications = pd.read_csv(f'{folder_path}/PRESCRIPTIONS.csv')

    # 3. read the other files
    df_DiagnosisICD  = pd.read_csv(f'{folder_path}/DIAGNOSES_ICD.csv')    # Diagnosis!
    df_ProceduresICD = pd.read_csv(f'{folder_path}/PROCEDURES_ICD.csv')    # Procedures!

    # 4. selected the data for the selected patients only...
    print('Use the patients inside the new DF....')
    new_Diagnosis  = df_DiagnosisICD[df_DiagnosisICD['SUBJECT_ID'].isin(patients)]
    new_Procedures = df_ProceduresICD[df_ProceduresICD['SUBJECT_ID'].isin(patients)]
    new_Medication = df_Medications[df_Medications['subject_id'].isin(patients)]


    # processing the data, and modify length of the diagnosis and procedure:
    print('For the given diagnosis, extract the sub dataframe....')
    new_Diagnosis.dropna(subset=['ICD9_CODE'], inplace=True)
    new_Procedures.dropna(subset=['ICD9_CODE'], inplace=True)
    new_Medication.dropna(subset=['drug'], inplace=True)

    new_Diagnosis['ICD9_CODE']  = new_Diagnosis['ICD9_CODE'].apply(extract3)
    new_Procedures['ICD9_CODE'] = new_Procedures['ICD9_CODE'].apply(extract2)
    # ----------------------------------------------------------------------------
    
    diag_frequency = new_Diagnosis['ICD9_CODE'].value_counts().head(num_Diseases).index.tolist()
    
    print('Diagnoses frequency = ', diag_frequency)
    
    new_Diagnosis  = new_Diagnosis[new_Diagnosis['ICD9_CODE'].isin(diag_frequency)]
    
    
    # ----------------------------------------------------------------------------
    # extracting the unique sets of nodes of diff category.
    Procedures = sorted(new_Procedures['ICD9_CODE'].unique())
    Medication = sorted(new_Medication['drug'].unique())
    Diagnosis  = new_Diagnosis['ICD9_CODE'].unique()

    print('General Information:\n---------------------------')
    print(f'Number of Patients = {len(patients)}')
    print(f'Number of Diagnosis = {len(Diagnosis)}')
    print(f'Number of procedures = {len(Procedures)}')
    print(f'Number of Medication = {len(Medication)}')

    return new_Diagnosis, new_Medication, new_Procedures



def getDict2(df, id1, id2, c1, c2):
    new_df = df[[id1, id2]].copy()    
    
    # Add the prefixes to each column
    new_df.loc[:, id1] = c1 + '_' + new_df[id1].astype(str)
    new_df.loc[:, id2] = c2 + '_' + new_df[id2].astype(str)
    
    # Remove duplicate rows
    new_df = new_df.drop_duplicates()
    return new_df


def getEdges(data, id1, id2):
    # Check if data is a DataFrame and extract edges accordingly
    if isinstance(data, pd.DataFrame):
        # Extract edges from the DataFrame
        EdgesList = list(data[[id1, id2]].itertuples(index=False, name=None))
    else:
        # Assuming data is a list of dictionaries
        EdgesList = [(d[id1], d[id2]) for d in data]
    
    return EdgesList


def get_homogeneous_graphs(new_Diagnosis, new_Prescriptions, new_Procedures):
    print('---------------------------------')
    print('Getting the Homogeneous graphs...')

    patient_visit_df    = getDict2(new_Diagnosis,  'SUBJECT_ID', 'HADM_ID', 'C', 'V')
    visit_diagnosis_df  = getDict2(new_Diagnosis, 'HADM_ID', 'ICD9_CODE',  'V', 'D')
    visit_procedure_df  = getDict2(new_Procedures, 'HADM_ID', 'ICD9_CODE', 'V', 'P')
    visit_medication_df = getDict2(new_Prescriptions, 'hadm_id', 'drug', 'V', 'M')

    # Edge Extractions
    CV_edges = getEdges(patient_visit_df, 'SUBJECT_ID', 'HADM_ID')
    VD_edges = getEdges(visit_diagnosis_df, 'HADM_ID', 'ICD9_CODE')
    VP_edges = getEdges(visit_procedure_df,'HADM_ID', 'ICD9_CODE')
    VM_edges = getEdges(visit_medication_df,  'hadm_id', 'drug')  

    print('Done --> Getting the Homogeneous graphs...')
    
    return   CV_edges , VD_edges , VP_edges , VM_edges



def get_Heterogeneous_graph(CV_edges , VD_edges , VP_edges , VM_edges):
    print('Creating the Heterogeneous graph...')

    edge_set = CV_edges + VD_edges + VP_edges + VM_edges #+ VI_edges

    tempG = nx.Graph()
    tempG.add_edges_from(edge_set)

    Nodes = list(tempG.nodes())
    N = len(Nodes)

    C_Nodes = [v for v in Nodes if v[0]=='C']
    V_Nodes = [v for v in Nodes if v[0]=='V']
    M_Nodes = [v for v in Nodes if v[0]=='M']
    D_Nodes = [v for v in Nodes if v[0]=='D']
    P_Nodes = [v for v in Nodes if v[0]=='P']

    print(f'number of patients = {len(C_Nodes)}')
    print(f'number of visits = {len(V_Nodes)}')
    print(f'number of Medication = {len(M_Nodes)}')
    print(f'number of Diagnoses = {len(D_Nodes)}')
    print(f'number of Procedures = {len(P_Nodes)}')

    The_Graph = nx.Graph()
    The_Graph.add_nodes_from(C_Nodes + V_Nodes + M_Nodes + D_Nodes + P_Nodes)
    The_Graph.add_edges_from(edge_set)

    print('Done --> Creating the Heterogeneous graph...')
    return The_Graph
    
    
    
    
    
# ==================================================================
def get_Nodes(G):
    # print('reading the nodes from the graph...')
    # G = nx.read_gml(file_path)    

    Nodes = list(G.nodes())

    C_Nodes = [v for v in Nodes if v[0]=='C']
    V_Nodes = [v for v in Nodes if v[0]=='V']
    M_Nodes = [v for v in Nodes if v[0]=='M']
    D_Nodes = [v for v in Nodes if v[0]=='D']
    P_Nodes = [v for v in Nodes if v[0]=='P']

    print(f'number of patients = {len(C_Nodes)}')
    print(f'number of visits = {len(V_Nodes)}')
    print(f'number of Medication = {len(M_Nodes)}')
    print(f'number of Diagnoses = {len(D_Nodes)}')
    print(f'number of Procedures = {len(P_Nodes)}')

    return sorted(C_Nodes), sorted(V_Nodes), sorted(M_Nodes), sorted(D_Nodes), sorted(P_Nodes), G

def read_embedding():
    
    print('Reading the embedding of the entire MIMIC dataset...')

    embedding_df = pd.read_csv(f'{folder_path}/grouped_emb2.csv')

    embedding_df = embedding_df.rename(columns={'SUBJECT_ID': 'Patient'})#, 'Embedding': 'Reduced_Embedding'})

    # convert the df to a dict of {patient: list of }

    patient_emb_dict = embedding_df.set_index('Patient')['Embedding'].to_dict()

    df_patients = embedding_df['Patient'].unique()

    print(f'number of patients in the embedding file is {len(df_patients)}')
    print('\tReturning the list of patients and their embedding...')
    return df_patients, patient_emb_dict, embedding_df
    
    
    
def proc_string_list(s):
    import re
    # Remove brackets and newline characters, then split based on spaces
    numbers_str = re.sub(r'[\[\]\n]', '', s).split()

    # Convert each element to float, then to int (by rounding)
    return [float(num) for num in numbers_str if num]



def mark_all_nodes(Nodes, C_Nodes, V_Nodes, M_Nodes, D_Nodes, P_Nodes):
    print('Assingning embedding to the other nodes.')
    print('\t The GNN model shall identify the other nodes as well, right?.')

    nodes_by_type = {'C':0, 'V':1, 'M': 2, 'D': 3, 'P':4}

    nodes_list = {'C': C_Nodes, 'V': V_Nodes, 'M': M_Nodes, 'D': D_Nodes, 'P': P_Nodes}

    emb = {}
    for n in Nodes:
        t = nodes_by_type[n[0]]
        emb[n] = [t, nodes_list[n[0]].index(n)]

    print('\tDone --> Assingning embedding to the other nodes.')
    return emb



def reading_pickle(n):
    with open(f'{n}', 'rb') as f:
        data = pd.read_pickle(f)
    # numpy_array = np.array(data)
    return data

def plot_dict_as_bar(dict_data, x_label, y_label, title):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Extract keys (items) and values (degrees) from the dictionary
    items = list(dict_data.keys())
    degrees = list(dict_data.values())

    # Create a bar plot
    plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
    plt.bar(items, degrees)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=90)  # Optional: Rotate x-axis labels for readability
    
    plt.tight_layout()  # Optional: Ensure labels fit within the figure boundaries
    plt.show()
    
    

def get_label(Nodes, G, Diagnosis_index):
    labels = []
    for node in Nodes:
        d = [0] * len(Diagnosis_index)
        if node[0] =='C':
            Visits = G.neighbors(node)
            node_diagnosis = []
            for v in Visits:
                node_diagnosis.extend([dd for dd in G.neighbors(v) if dd[0]=='D'])
            
            for dd in set(node_diagnosis):
                d[Diagnosis_index[dd]] = 1
        labels.append(d)
    return labels


def remove_one_class_columns(Y, num_Patients):
    def column_contains_one_class(column):
        unique_values = np.unique(column)  # Get unique values in the column
        return len(unique_values) == 1  # Column contains only one class if unique values are 1

    columns_to_keep = []

    # Iterate over each column in Y
    for column_index in range(Y.shape[1]):
        column = Y[:num_Patients, column_index]  # Extract the specific column
        if not column_contains_one_class(column):
            columns_to_keep.append(column_index)

    # Create a new array Y_new with only the columns that are not one-class
    Y_new = Y[:, columns_to_keep]

    return Y_new


    
    
# =========================================================================
def get_adjacency_matrix(G, Nodes1, Nodes2):
    # Ensure that Nodes1 and Nodes2 are lists or are castable to lists
    Nodes1 = list(Nodes1)
    Nodes2 = list(Nodes2)
    
    # Map nodes to indices in the matrix
    node_to_index = {node: i for i, node in enumerate(Nodes1 + Nodes2)}
    
    # Get adjacency matrix in sparse format
    sparse_matrix = nx.adjacency_matrix(G, nodelist=Nodes1+Nodes2)
    
    # Convert to dense format if needed
    dense_matrix = sparse_matrix.todense()
    
    # Extract submatrix corresponding to Nodes1 and Nodes2
    W = dense_matrix[:len(Nodes1), len(Nodes1):]
    
    return np.array(W)



def get_adjacency_matrix1(G, Nodes1, Nodes2):
    # Ensure that Nodes1 and Nodes2 are lists or are castable to lists
    Nodes1 = list(Nodes1)
    Nodes2 = list(Nodes2)
    
    # Get adjacency matrix in sparse format for the union of Nodes1 and Nodes2
    sparse_matrix = nx.adjacency_matrix(G, nodelist=Nodes1+Nodes2)
    
    # Map nodes to their respective indices in the adjacency matrix
    node_to_index = {node: i for i, node in enumerate(Nodes1 + Nodes2)}
    
    # Extract indices for Nodes1 and Nodes2
    indices1 = [node_to_index[node] for node in Nodes1 if node in node_to_index]
    indices2 = [node_to_index[node] for node in Nodes2 if node in node_to_index]
    
    # Extract submatrix corresponding to Nodes1 and Nodes2
    W = sparse_matrix[indices1, :][:, indices2]
    
    return W

# def M(W1, W2):
#     return np.dot(W1, W2)

def M(W1, W2):
    # If W1 and W2 are sparse matrices, you can convert them to CSR format
    print(f'multiplying {W1.shape} * {W2.shape}...')
    W1_csr = csr_matrix(W1)
    W2_csr = csr_matrix(W2)

    # Perform the multiplication
    result = W1_csr.dot(W2_csr)
    print("Done multiplication...")
    return result

def symmetric_assign(W, shift, t):
    '''positioning W into the right t place'''
    rows = W.shape[0]
    cols = W.shape[1]
    
    newW = np.zeros((t,t))
    for i in range(0, rows):
        for j in range(0, cols):
            newW[i+shift][j+shift] = W[i][j]
    return newW

def symmetric_assign2(W, shift, t):
    '''positioning W into the right t place'''
    rows, cols = W.shape
    # Initialize a larger matrix with zeros
    newW = np.zeros((t, t), dtype=np.float32)
    
    print(rows, cols, newW.shape)

    # Assign W into newW at the specified shift
    newW[shift:shift + rows, shift:shift + cols] = W.toarray()

    return newW

def symmetric_assign_Coo(W, shift, t):
    # Create a LIL matrix for efficient assignment
    newW = lil_matrix((t, t), dtype=np.float32)
    
    # Find the indices of non-zero elements in W
    non_zero_indices = np.nonzero(W)
    rows, cols = non_zero_indices
    print(rows, cols)
    # Iterate over the non-zero elements of W using the indices
    for i, j in zip(rows, cols):
        value = W[i, j]
        # Add the value at the shifted position
        newW[shift + i, shift + j] = value

    print("symmmetric_assign_Coo is complete...")
    return newW

def norm_max(W):
    # Normalizing the array    
    max_value = np.max(W)
    print(f'{W.shape}\t{max_value}')
    return W / max_value

def asymmetric_assign_Coo(W, shift_row, shift_col, t):
    # Create a LIL matrix for efficient assignment
    newW = lil_matrix((t, t), dtype=np.float32)
    
    # Find the indices of non-zero elements in W
    non_zero_indices = np.nonzero(W)
    rows, cols = non_zero_indices

    # Iterate over the non-zero elements of W using the indices
    for i, j in zip(rows, cols):
        value = W[i, j]
        # Add the value at the shifted position
        newW[shift_row + i, shift_col + j] = value
        # Assuming you want a symmetric assignment
        newW[shift_col + j, shift_row + i] = value
        
    return newW

# =========================================================================

def get_edges_dict(the_path):
    A = scipy.sparse.load_npz(the_path)
    if not isinstance(A, scipy.sparse.coo_matrix):
        A = A.tocoo()
    filtered_entries = (A.col > A.row) & (A.data > 0)
    upper_triangle_positive = {(row, col): data for row, col, data in zip(A.row[filtered_entries], A.col[filtered_entries], A.data[filtered_entries])}
    return upper_triangle_positive

# ==========================================================================


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    