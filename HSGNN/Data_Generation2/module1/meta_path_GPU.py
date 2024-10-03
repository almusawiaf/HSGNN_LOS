from scipy.sparse import csr_matrix, vstack
from joblib import Parallel, delayed

import networkx as nx
import numpy as np
import pickle
import scipy
from scipy import sparse
from scipy.sparse import issparse
import os        

import cupy as cp
import time

    

def save_list_as_pickle(L, given_path, file_name):
    import pickle
    print(f'saving to {given_path}/{file_name}.pkl')
    with open(f'{given_path}/{file_name}.pkl', 'wb') as file:
        pickle.dump(L, file)

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict  
        

class Meta_path:
    ''' The goal of this class is to read a heterogeneous graph HG, 
        type of similarity metrics 1. PC: Path-Count, 2. SPS: Symmetric PathSim,
        and create the list of meta-path-based similarities. 
        Then, we saved them to the given saving_path.'''
    
    def __init__(self, HG, similarity_type = 'PC', saving_path = ''):
        self.HG = HG
        self.saving_path = saving_path

        Nodes = list(self.HG.nodes())

        self.Patients =    [v for v in Nodes if v[0]=='C']
        self.Visits =      [v for v in Nodes if v[0]=='V']
        self.Medications = [v for v in Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in Nodes if v[0]=='D']
        self.Procedures =  [v for v in Nodes if v[0]=='P']
        self.Labs       =  [v for v in Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in Nodes if v[0]=='B']
        self.Nodes = Nodes
        
        print('extracting As from HG\n')
        W_cv = self.subset_adjacency_matrix(self.Patients + self.Visits)
        W_vm = self.subset_adjacency_matrix(self.Visits + self.Medications)
        W_vd = self.subset_adjacency_matrix(self.Visits + self.Diagnosis)
        W_vp = self.subset_adjacency_matrix(self.Visits + self.Procedures)          
        W_vl = self.subset_adjacency_matrix(self.Visits + self.Labs)  
        W_vb = self.subset_adjacency_matrix(self.Visits + self.MicroBio) 
        
        self.save_sparse(W_cv, 1)
        self.save_sparse(W_vm, 2)
        self.save_sparse(W_vd, 3)
        self.save_sparse(W_vp, 4)
        self.save_sparse(W_vl, 5)
        self.save_sparse(W_vb, 6)                 

        print('=============================================================')
        print('Multiplication phase...\n')
        # PathCount 
        # Heterogeneous similarity
        
        self.M_gpu(W_vm, W_vm.T, 7) # W_VMV
        self.M_gpu(W_vd, W_vd.T, 8) # W_VDV
        self.M_gpu(W_vp, W_vp.T, 9) # W_VPV
        self.M_gpu(W_vl, W_vl.T, 10) # W_VLV
        self.M_gpu(W_vb, W_vb.T, 11) # W_VBV
               

        self.M_gpu(W_cv, W_vm, 12)
        W_CVM = self.load_sparse(12)
        self.M_gpu(W_CVM, W_CVM.T, 13) # CVMVC
        del W_CVM

        self.M_gpu(W_cv, W_vd, 14)
        W_CVD = self.load_sparse(14)
        self.M_gpu(W_CVD, W_CVD.T, 15) # CVDVC
        del W_CVD

        self.M_gpu(W_cv, W_vp, 16)
        W_CVP = self.load_sparse(16)
        self.M_gpu(W_CVP, W_CVP.T, 17) # CVPVC
        del W_CVP

        self.M_gpu(W_cv, W_vl, 18)
        W_CVL = self.load_sparse(18)
        self.M_gpu(W_CVL, W_CVL.T, 19) # CVLVC
        del W_CVL

        self.M_gpu(W_cv, W_vb, 20)
        W_CVB = self.load_sparse(20)
        self.M_gpu(W_CVB, W_CVB.T, 21) # CVBVC
        del W_CVB
        del W_cv
        
        
        self.M_gpu(W_vd.T, W_vm, 22)
        W_DVM = self.load_sparse(22)
        self.M_gpu(W_DVM, W_DVM.T, 23) # W_DVMVD
        self.M_gpu(W_DVM.T, W_DVM, 24) # W_MVDVM
        del W_DVM

        self.M_gpu(W_vd.T, W_vp, 25)
        W_DVP = self.load_sparse(25)
        self.M_gpu(W_DVP, W_DVP.T, 26) # W_DVPVD
        self.M_gpu(W_DVP.T, W_DVP, 27) # W_PVDVP
        del W_DVP
        
        
        self.M_gpu(W_vd.T, W_vl, 28)
        W_DVL = self.load_sparse(28)
        self.M_gpu(W_DVL, W_DVL.T, 29) # W_DVLVD 
        self.M_gpu(W_DVL.T, W_DVL, 30) # W_LVDVL
        del W_DVL
        
        
        self.M_gpu(W_vd.T, W_vb, 31)
        W_DVB = self.load_sparse(31)
        self.M_gpu(W_DVB, W_DVB.T, 32) # W_DVBVD
        self.M_gpu(W_DVB.T, W_DVB, 33) # W_BVDVB
        del W_DVB        
        del W_vd
        

        self.M_gpu(W_vp.T, W_vm, 34)
        W_PVM = self.load_sparse(34)
        self.M_gpu(W_PVM.T, W_PVM, 35) # W_MVPVM 
        self.M_gpu(W_PVM, W_PVM.T, 36) # W_PVMVP
        del W_PVM

        self.M_gpu(W_vp.T, W_vl, 37)
        W_PVL = self.load_sparse(37)
        self.M_gpu(W_PVL, W_PVL.T, 38) # W_PVLVP
        self.M_gpu(W_PVL.T, W_PVL, 39) # W_LVPVL 
        del W_PVL

        self.M_gpu(W_vp.T, W_vb, 40)
        W_PVB = self.load_sparse(40)
        self.M_gpu(W_PVB, W_PVB.T, 41) # W_PVBVP
        self.M_gpu(W_PVB.T, W_PVB, 42) # W_BVPVB
        del W_PVB 
        del W_vp

        
        self.M_gpu(W_vm.T, W_vl, 43)
        W_MVL = self.load_sparse(43)
        self.M_gpu(W_MVL, W_MVL.T, 44) # W_MVLVM
        self.M_gpu(W_MVL.T, W_MVL, 45) # W_LVMVL
        del W_MVL

        self.M_gpu(W_vm.T, W_vb, 46)
        W_MVB = self.load_sparse(46)
        self.M_gpu(W_MVB, W_MVB.T, 47) # W_MVBVM
        self.M_gpu(W_MVB.T, W_MVB, 48) # W_BVMVB
        del W_MVB
        del W_vm

        
        self.M_gpu(W_vl.T, W_vb, 49)
        W_LVB = self.load_sparse(49)
        self.M_gpu(W_LVB, W_LVB.T, 50) # W_LVBVL
        self.M_gpu(W_LVB, W_LVB.T, 51) # W_BVLVB
        del W_LVB
        del W_vl
        del W_vb
        print('=============================================================')
        print('Multiplication phase completed!\n')
        save_list_as_pickle([i for i in range(52)], f"{saving_path}/As", 'selected_i')


    def save_sparse(self, A, i):
        if not issparse(A):
            A = sparse.csr_matrix(A)
        sparse.save_npz(f"{self.saving_path}/As/sparse_matrix_{i}.npz", A)
    
    def load_sparse(self, i):
        file_path = f"{self.saving_path}/As/sparse_matrix_{i}.npz"
        return sparse.load_npz(file_path)

    def subset_adjacency_matrix(self, subset_nodes):
        adj_matrix = nx.to_numpy_array(self.HG)
        mask = np.isin(self.Nodes, subset_nodes)
        for i in range(len(self.Nodes)):
            if not mask[i]:
                adj_matrix[i, :] = 0  # Zero out the row
                adj_matrix[:, i] = 0  # Zero out the column
        
        return adj_matrix

    # Function to multiply matrices on GPU using CuPy
    def M_gpu(self, A, B, i):
        """
        Multiplies two matrices A and B using GPU with CuPy.

        Parameters:
        A : ndarray or sparse matrix
            First matrix.
        B : ndarray or sparse matrix
            Second matrix.

        Returns:
        result : ndarray
            Resulting matrix after multiplication.
        """
        if sparse.issparse(A):
            A = A.toarray()  # Convert sparse to dense NumPy array
        if sparse.issparse(B):
            B = B.toarray()  # Convert sparse to dense NumPy array
    
        # Convert matrices to GPU arrays (CuPy arrays)
        A_gpu = cp.array(A)  # Automatically transfers A to GPU
        B_gpu = cp.array(B)  # Automatically transfers B to GPU

        # Start the timer
        cp.cuda.Stream.null.synchronize()  # Ensure all previous GPU tasks are complete
        start_time = time.time()

        # Perform matrix multiplication on the GPU
        result_gpu = cp.dot(A_gpu, B_gpu)

        # Synchronize again to ensure the multiplication is done before measuring the time
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()

        # Transfer the result back to the CPU as a NumPy array
        A = cp.asnumpy(result_gpu)

        # Print the time taken
        print(f"Matrix multiplication on GPU took {end_time - start_time:.6f} seconds")
        self.save_sparse(A, i)



# The rest of the code remains unchanged
def create_folder(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path)
