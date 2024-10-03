from scipy.sparse import csr_matrix, vstack
from joblib import Parallel, delayed

import networkx as nx
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import pickle
import scipy
from scipy import sparse
from scipy.sparse import issparse
import os        


def save_list_as_pickle(L, given_path, file_name):
    import pickle
    print(f'saving to {given_path}/{file_name}.pkl')
    with open(f'{given_path}/{file_name}.pkl', 'wb') as file:
        pickle.dump(L, file)

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict  
        
        

class Reduction:
    ''' The goal of this class is to:
        1. read a list of similarity matrices.
        2. select the highest weighted edges
        3. convert them to edge_list
        4. save the final edge_weight per matrix.'''
        
    def __init__(self, saving_path = '', gpu = False):
                        
        Ws1 = self.read_Ws(saving_path, 'As')
        # self.Ws2 = self.read_Ws(saving_path, 'Cosine_As')
        Ws = self.selecting_high_edges(Ws1)
        self.final_Ws_with_unique_list_of_edges(Ws, saving_path)        
        
    def read_Ws(self, saving_path, folder_name):
        selected_i = load_dict_from_pickle(f"{saving_path}/{folder_name}/selected_i.pkl")
        return [self.get_edges_dict(f'{saving_path}/{folder_name}/sparse_matrix_{i}.npz') for i in selected_i]

    def get_edges_dict(self, the_path):
        A = sparse.load_npz(the_path)
        if not isinstance(A, scipy.sparse.coo_matrix):
            A = A.tocoo()
        filtered_entries = (A.col > A.row) & (A.data > 0)
        upper_triangle_positive = {(row, col): data for row, col, data in zip(A.row[filtered_entries], A.col[filtered_entries], A.data[filtered_entries])}
        return upper_triangle_positive

    def selecting_high_edges(self, Ws):
        '''Selecting top weighted edges'''
        D = []
        for i, A in enumerate(Ws):
            # A is a dictionary, no need to check for sparsity
            num_non_zeros = len(A)
            print(f"Matrix {i}: {num_non_zeros} non-zero elements")
            
            if num_non_zeros > 0:
                if num_non_zeros > 1000000:
                    B = keep_top_million(A, top_n=1000000)  # Define this function to keep top N edges in the dictionary
                    print(f"\tSaving one million non-zero values... (after reduction: {len(B)} non-zero elements)")
                else:
                    B = A
                    print(f"\tSaving all non-zero values... ({num_non_zeros} non-zero elements)")
            
                D.append(B)  # Store the dictionary directly
            else:
                print(f'Matrix {i} has zero values. Not saving...')
        
        return D


    def final_Ws_with_unique_list_of_edges(self, D, saving_path):        
        '''creating and saving the last representation of the edges_list and edges_weight'''
        sorted_list_of_dicts = sorted(D, key=lambda x: len(x), reverse=True)        
        unique_edges = set()        
        for e in sorted_list_of_dicts:
            unique_edges.update(e.keys())
        
        create_folder(f'{saving_path}/edges')
        with open(f'{saving_path}/edges/edge_list.pkl', 'wb') as file:
            pickle.dump(unique_edges, file)
        
        print('done saving [unique edges]: ', len(unique_edges))        

        # ======================================== Reflect the (edge_list) into all A's =======================================
        for i, d in enumerate(D):
            print(f'Working on {i}th file...')
            results = []
            for e in unique_edges:
                if e in d:
                    results.append(d[e])
                else:
                    results.append(0)
                    
            with open(f'{saving_path}/edges/edge_weight{i}.pkl', 'wb') as file:
                pickle.dump(results, file)
        
            
    
    
    def get_SNF(self):
        SNF_inst = SNF_class(self.HG, self.Nodes)
        return SNF_inst.A


def create_folder(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path)


def keep_top_million(edge_dict, top_n=1000000):
    if len(edge_dict) <= top_n:
        return edge_dict
    
    # Step 1: Sort the dictionary by values (edge weights) and keep only the top N
    sorted_edges = sorted(edge_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Step 2: Keep the top N edges
    top_edges = dict(sorted_edges[:top_n])
    
    return top_edges

# def keep_top_million(sparse_matrix, top_n=1000000):
#     # Convert to CSR if the matrix is dense
#     if isinstance(sparse_matrix, np.ndarray):
#         sparse_matrix = sparse.csr_matrix(sparse_matrix)

#     # Step 1: Find the threshold for the top million values
#     if sparse_matrix.nnz <= top_n:
#         # If the matrix has fewer non-zeros than top_n, return it as is
#         return sparse_matrix
    
#     # Extract the non-zero data from the sparse matrix
#     sorted_data = np.sort(sparse_matrix.data)[-top_n]
    
#     # Step 2: Filter the sparse matrix based on this threshold
#     mask = sparse_matrix.data >= sorted_data
    
#     # Apply the mask to keep only values in the top million
#     filtered_data = sparse_matrix.data[mask]
#     filtered_indices = sparse_matrix.indices[mask]
#     filtered_indptr = np.zeros(sparse_matrix.shape[0] + 1, dtype=int)
    
#     # Recalculate indptr array based on the filtered data
#     current_index = 0
#     for i in range(sparse_matrix.shape[0]):
#         row_start = sparse_matrix.indptr[i]
#         row_end = sparse_matrix.indptr[i+1]
        
#         # Count non-zero elements in the current row that are in the top million
#         filtered_indptr[i+1] = filtered_indptr[i] + np.sum(mask[row_start:row_end])
    
#     # Create a new sparse matrix with the filtered data
#     filtered_matrix = sparse.csr_matrix((filtered_data, filtered_indices, filtered_indptr), shape=sparse_matrix.shape)
    
#     return filtered_matrix



    
