from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np


class Patients_Similarity:
    
    def __init__(self, HG, Nodes):
        '''reading a HG and Nodes
        1. create OHV per node type
        2. measure the similarity
        3. measure SNF and hold it as A.
        '''
        self.HG = HG
        self.Nodes = Nodes
        # ======================================================
        self.Patients =    [v for v in self.Nodes if v[0]=='C']
        self.Visits =      [v for v in self.Nodes if v[0]=='V']
        self.Medications = [v for v in self.Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in self.Nodes if v[0]=='D']
        self.Procedures =  [v for v in self.Nodes if v[0]=='P']
        self.Labs       =  [v for v in self.Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in self.Nodes if v[0]=='B']
        # ======================================================
        D = self.get_X('D')
        M = self.get_X('M')
        P = self.get_X('P')
        L = self.get_X('L')
        B = self.get_X('B')
        # ======================================================
        print('Measure the similarity')
        DB = [cosine_similarity(X) for X in [D, M, P, L, B]]

        self.A, _, _ = self.SNF(DB, 'euclidean')
        self.expand_A()
        
        
        
    def get_X(self, clinical_type):
        
        print(f'Getting the OHV for {clinical_type}')
        if clinical_type=='M':
            F = self.Medications
        elif clinical_type=='P':
            F = self.Procedures
        elif clinical_type=='L':
            F = self.Labs
        elif clinical_type=='D':
            F = self.Diagnosis
        elif clinical_type=='B':
            F = self.MicroBio
            
        F_indeces = {p:k for k,p in enumerate(F)}

        X = []
        for v in self.Patients:
            f = [0] * len(F)
            for u_visit in self.HG.neighbors(v):
                for u in self.HG.neighbors(u_visit):
                    if u[0] in [clinical_type]:
                        f[F_indeces[u]] = 1
            X.append(f)
        
        return np.array(X)

    def get_X_sub_case(self, clinical_type):
        
        print(f'Getting the OHV for {clinical_type}')
        if clinical_type=='G':
            F = self.Gender
        elif clinical_type=='E':
            F = self.Expire_Flag
            
        F_indeces = {p:k for k,p in enumerate(F)}

        X = []
        for v in self.Patients:
            f = [0] * len(F)
            for u in self.HG.neighbors(v):
                if u[0] in [clinical_type]:
                    f[F_indeces[u]] = 1
            X.append(f)
        
        return np.array(X)
    
    def expand_A(self):
        n = len(self.Patients)
        m = len(self.Nodes)
        expanded_matrix = np.zeros((m, m))
        expanded_matrix[:n, :n] = self.A
        
        self.A = expanded_matrix
