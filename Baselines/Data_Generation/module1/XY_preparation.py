'''
generate_HG is class used to generate the 203 heterogeneous graph only.
We only included patients with diagnoses.
'''
import numpy as np
import networkx as nx
import pandas as pd

# Extracting the X and Y data using the passed HG graph.

class XY_preparation:

    
    def __init__(self, HG):
        self.HG = HG
        self.folder_path = '/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources'

        Nodes = list(self.HG.nodes())
        self.Patients =    [v for v in Nodes if v[0]=='C']
        self.Visits =      [v for v in Nodes if v[0]=='V']
        self.Medications = [v for v in Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in Nodes if v[0]=='D']
        self.Procedures =  [v for v in Nodes if v[0]=='P']
        self.Labs       =  [v for v in Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in Nodes if v[0]=='B']
        self.Nodes = Nodes
        
        self.X_visit = self.get_X_visit_level()
        self.Y_visit = self.get_Y_visit_level()
        
    def get_X_visit_level(self):
        print('getting the feature set for all nodes: visit_level')
        F = self.Diagnosis + self.Medications  + self.Procedures + self.Labs + self.MicroBio 
        F_indeces = {p:k for k,p in enumerate(F)}

        X = []
        for v in self.Visits:
            f = [0] * len(F)
            for u in self.HG.neighbors(v):
                if u[0] in ['P', 'M', 'L', 'B', 'D']:
                    f[F_indeces[u]] = 1
            X.append(f)
        
        return np.array(X)
    
    
    

    def get_Y_visit_level(self):
        '''
        1. read the ADMISSIONS.csv file
        2. include rows with HADM_ID in self.Visits
        3. extract the contineous value LOS associated to the visit.
        4. convert and save the bucketized LOS with the corresponding visit.
        5. save it to self.Y
        '''

        LOS_DF = self.get_HADM_ID_LOS()
        print(LOS_DF.head(5))
        temp_Y = []
        for node in self.Visits:
            los_value = LOS_DF[LOS_DF['HADM_ID'] == int(node[2:])]['LOS']
            y = 0
            if not los_value.empty:
                y = self.classify_los(los_value.iloc[0])
            temp_Y.append(y)        
            
        return np.array(temp_Y)
            

    def get_HADM_ID_LOS(self):
        df_admissions = pd.read_csv(f'{self.folder_path}/ADMISSIONS.csv')

        # Ensure ADMITTIME and DISCHTIME are in datetime format
        df_admissions['ADMITTIME'] = pd.to_datetime(df_admissions['ADMITTIME'])
        df_admissions['DISCHTIME'] = pd.to_datetime(df_admissions['DISCHTIME'])
        
        # Calculate the Length of Stay (LOS) in days
        df_admissions['LOS'] = (df_admissions['DISCHTIME'] - df_admissions['ADMITTIME']).dt.total_seconds() / (24 * 60 * 60)
        
        return df_admissions[['HADM_ID', 'LOS']]

    def classify_los(self, los):
        if los < 3:
            return 0
        elif 3 <= los <= 7:
            return 1
        elif los > 5:
            return 2

