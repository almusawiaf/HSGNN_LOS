'''
generate_HG is class used to generate the 203 heterogeneous graph only.
We only included patients with diagnoses.
'''
import numpy as np
import networkx as nx

class XY_preparation:

    
    def __init__(self, HG):
        self.HG = HG
        
        Nodes = list(self.HG.nodes())
        self.Patients =    [v for v in Nodes if v[0]=='C']
        self.Visits =      [v for v in Nodes if v[0]=='V']
        self.Medications = [v for v in Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in Nodes if v[0]=='D']
        self.Procedures =  [v for v in Nodes if v[0]=='P']
        self.Labs       =  [v for v in Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in Nodes if v[0]=='B']
        self.Nodes = Nodes
        
        self.X = self.get_X()
        self.Y = self.remove_one_class_columns(self.get_Y())

        self.X_visit = self.get_X_visit_level()
        self.Y_visit = self.remove_one_class_columns(self.get_Y_visit_level())
        
    def get_X(self):
        print('getting the feature set for all nodes')
        F = self.Medications  + self.Procedures + self.Labs + self.MicroBio 
        F_indeces = {p:k for k,p in enumerate(F)}

        X = []
        for v in self.Nodes:
            f = [0] * len(F)
            if v[0]=='C':
                for u_visit in self.HG.neighbors(v):
                    for u in self.HG.neighbors(u_visit):
                        if u[0] in ['P', 'M', 'L', 'B']:
                            f[F_indeces[u]] = 1
            X.append(f)
        
        return np.array(X)
    
    
    

    def get_Y(self):
        '''return a binary matrix of diagnoses, if exites across all visits.'''

        def and_over_rows(binary_lists):
            """Compute the AND function across all rows of a list of lists of binary numbers."""
            binary_array = np.array(binary_lists)
            return np.all(binary_array == 1, axis=0).astype(int).tolist()          
            
        F = self.Diagnosis
        F_indeces = {p:k for k,p in enumerate(F)}

        Y = []
        for v in self.Nodes:
            if v[0]=='C':
                temp_Y = []
                for u_visit in self.HG.neighbors(v):
                    f = [0] * len(F)
                    for u in self.HG.neighbors(u_visit):
                        if u[0] in ['D']:
                            f[F_indeces[u]] = 1
                    temp_Y.append(f)
                
                Y.append(and_over_rows(temp_Y))
            else:
                Y.append([0] * len(F))
        
        return np.array(Y)
    
    
    

    def get_X_visit_level(self):
        print('getting the feature set for all nodes: visit_level')
        F = self.Medications  + self.Procedures + self.Labs + self.MicroBio 
        F_indeces = {p:k for k,p in enumerate(F)}

        X = []
        for v in self.Nodes:
            f = [0] * len(F)
            if v[0]=='V':
                for u in self.HG.neighbors(v):
                    if u[0] in ['P', 'M', 'L', 'B']:
                        f[F_indeces[u]] = 1
            X.append(f)
        
        return np.array(X)
    
    
    

    def get_Y_visit_level(self):
        F = self.Diagnosis
        F_indeces = {p:k for k,p in enumerate(F)}

        Y = []
        for v in self.Nodes:
            f = [0] * len(F)
            if v[0]=='V':
                for u in self.HG.neighbors(v):
                    if u[0] in ['D']:
                        f[F_indeces[u]] = 1
            Y.append(f)
        
        return np.array(Y)
    
    
    
    
                
    def remove_one_class_columns(self, Y):
        def column_contains_one_class(column):
            unique_values = np.unique(column)  # Get unique values in the column
            return len(unique_values) == 1  # Column contains only one class if unique values are 1
    
        columns_to_keep = []
        num_Patients = len(self.Patients)
    
        # Iterate over each column in Y
        for column_index in range(Y.shape[1]):
            column = Y[:num_Patients, column_index]  # Extract the specific column
            if not column_contains_one_class(column):
                columns_to_keep.append(column_index)
    
        # Create a new array Y_new with only the columns that are not one-class
        Y_new = Y[:, columns_to_keep]
    
        return Y_new

    











