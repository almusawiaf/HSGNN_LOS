'''
generate_HG is class used to generate the 203 heterogeneous graph only.
We only included patients with diagnoses.
'''
import pandas as pd
import networkx as nx
from copy import deepcopy

'''NO DEMOGRAPHICS INFORMATION ADDED'''

class Generate_HG:
    
    def __init__(self, folder_path):

        self.folder_path = folder_path
        
        print('Loading the dataframes...')
        
        new_Diagnosis, new_Prescriptions, new_Procedures, new_LabTest, new_MicroBio = self.load_patients_data()
        
        print('Extracting bipartite networks...')


        CV = self.get_Bipartite(new_Diagnosis,    'SUBJECT_ID', 'HADM_ID',  'C', 'V', 'Visits')
        VD = self.get_Bipartite(new_Diagnosis,    'HADM_ID', 'ICD9_CODE',   'V', 'D', 'Diagnosis')
        VP = self.get_Bipartite(new_Procedures,   'HADM_ID', 'ICD9_CODE',   'V', 'P', 'Procedures')
        VM = self.get_Bipartite(new_Prescriptions,'hadm_id', 'drug',        'V', 'M', 'Medications')
        VL = self.get_Bipartite(new_LabTest,      'HADM_ID', 'ITEMID_FLAG', 'V', 'L', 'Lab tests')
        VB = self.get_Bipartite(new_MicroBio,     'HADM_ID', 'SPEC_ITEMID', 'V', 'B', 'MicroBiology tests')
        
        self.HG = nx.Graph()
        edges_list = CV + VD + VP + VM + VB + VL      
                
        self.HG.add_edges_from(edges_list)
        G_statistics(self.HG)
        
        self.selecting_top_labs()       
        self.remove_isolated_nodes()        
        self.update_statistics()

        G_statistics(self.HG)

        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    def update_statistics(self):
        Nodes = list(self.HG.nodes())
        self.Patients =    [v for v in Nodes if v[0]=='C']
        self.Visits =      [v for v in Nodes if v[0]=='V']
        self.Medications = [v for v in Nodes if v[0]=='M']
        self.Diagnoses  =  [v for v in Nodes if v[0]=='D']
        self.Procedures =  [v for v in Nodes if v[0]=='P']
        self.Labs       =  [v for v in Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in Nodes if v[0]=='B']
        self.Nodes = self.Patients  + self.Visits + self.Medications + self.Diagnoses + self.Procedures + self.Labs + self.MicroBio
        

    def get_Bipartite(self, DF, id1, id2, c1, c2, msg):
        '''DF: dataframe, id1: row1, id2: row2, c1, c2: node code'''  
        print(f'\nExtracting and adding data of {msg}')

        DF2    = self.getDict2(DF,  id1, id2, c1, c2)
        return self.getEdges(DF2, id1, id2)
    
    def split_lab_test(self, lab_df):
        print('Splitting lab tests')
        # Step 1: Fill NaN values in the 'FLAG' column with 'normal'
        lab_df['FLAG'] = lab_df['FLAG'].fillna('normal')
        
        # Step 2: Remove rows where 'FLAG' equals 'delta'
        lab_df = lab_df[lab_df['FLAG'] != 'delta']
        
        # Step 3: Create a new DataFrame with HADM_ID and a concatenated column 'itemid_flag'
        # Concatenate 'ITEMID' and 'FLAG' as strings
        lab_df.loc[:, 'ITEMID_FLAG'] = lab_df['ITEMID'].astype(str) + '_' + lab_df['FLAG'].astype(str)

        
        # Create the new DataFrame with 'HADM_ID' and the concatenated 'itemid_flag' column
        new_df = lab_df[['HADM_ID', 'ITEMID_FLAG']].copy()
        print(f'Number of visits here is {len(new_df["HADM_ID"].unique())}')

        return new_df

    def remove_isolated_nodes(self):
        print('Removing isolated nodes')
        self.update_statistics()
        isolated_nodes = [v for v in self.Nodes if self.HG.degree(v)==0]
        self.HG = remove_patients_and_linked_visits(isolated_nodes, self.HG)


    def extract3(self, code):
        return str(code)[:3]
    
    def extract2(self, code):
        return str(code)[:2]
    
    def load_patients_data(self):
        # Loading the data
        df_Medications   = pd.read_csv(f'{self.folder_path}/PRESCRIPTIONS.csv')
        df_DiagnosisICD  = pd.read_csv(f'{self.folder_path}/DIAGNOSES_ICD.csv')    # Diagnosis!
        df_ProceduresICD = pd.read_csv(f'{self.folder_path}/PROCEDURES_ICD.csv')    # Procedures!
        df_labs          = pd.read_csv(f'{self.folder_path}/LABEVENTS.csv')    # Lab test!
        df_microbio      = pd.read_csv(f'{self.folder_path}/MICROBIOLOGYEVENTS.csv')    # Microbiology!
        
        
        # Handling missing values upfront (dropping rows with missing important columns)
        df_DiagnosisICD.dropna(subset=['HADM_ID', 'ICD9_CODE'], inplace=True)
        df_ProceduresICD.dropna(subset=['ICD9_CODE'], inplace=True)
        df_Medications.dropna(subset=['drug'], inplace=True)
        df_labs.dropna(subset=['HADM_ID'], inplace=True)
        df_labs.dropna(subset=['ITEMID'], inplace=True)
        df_microbio.dropna(subset=['ORG_ITEMID'], inplace=True)
        
        # Extract unique visits and patients from the diagnosis DataFrame
        visits = df_DiagnosisICD['HADM_ID'].unique()
        patients = df_DiagnosisICD['SUBJECT_ID'].unique()

        df_labs['HADM_ID'] = df_labs['HADM_ID'].astype(int)

        df_labs = self.split_lab_test(df_labs)        

        # Filtering the data for selected patients and visits
        print('Use the patients inside the new DataFrame....')
        new_Diagnosis = df_DiagnosisICD[df_DiagnosisICD['HADM_ID'].isin(visits)].copy()
        new_Procedures = df_ProceduresICD[df_ProceduresICD['HADM_ID'].isin(visits)].copy()
        new_Medication = df_Medications[df_Medications['hadm_id'].isin(visits)].copy()
        new_LabTest = df_labs[df_labs['HADM_ID'].isin(visits)].copy()
        new_MicroBio = df_microbio[df_microbio['HADM_ID'].isin(visits)].copy()
        
        print('Dropping NaN visits')
        new_Diagnosis.dropna(subset=['HADM_ID'], inplace=True)
        new_Procedures.dropna(subset=['HADM_ID'], inplace=True)
        new_Medication.dropna(subset=['hadm_id'], inplace=True)
        new_LabTest.dropna(subset=['HADM_ID'], inplace=True)
        new_MicroBio.dropna(subset=['HADM_ID'], inplace=True)
    
        new_Diagnosis['ICD9_CODE']  = new_Diagnosis['ICD9_CODE'].apply(self.extract3)
        new_Procedures['ICD9_CODE'] = new_Procedures['ICD9_CODE'].apply(self.extract2)
        # ----------------------------------------------------------------------------
        
        diag_frequency = new_Diagnosis['ICD9_CODE'].value_counts().head(203).index.tolist()        
        new_Diagnosis  = new_Diagnosis[new_Diagnosis['ICD9_CODE'].isin(diag_frequency)]        
        
        # ----------------------------------------------------------------------------
        # extracting the unique sets of nodes of diff category.
        Procedures = sorted(new_Procedures['ICD9_CODE'].unique())
        Medication = sorted(new_Medication['drug'].unique())
        Diagnosis  = new_Diagnosis['ICD9_CODE'].unique()
        LabTests   = new_LabTest['ITEMID_FLAG'].unique()
        MicroBio   = new_MicroBio['SPEC_ITEMID'].unique()
    
        print('General Information:\n---------------------------')
        print(f'Number of Patients = {len(patients)}')
        print(f'Number of Visits = {len(visits)}')
        print(f'Number of Diagnosis = {len(Diagnosis)}')
        print(f'Number of procedures = {len(Procedures)}')
        print(f'Number of Medication = {len(Medication)}')
        print(f'Number of Lab tests  = {len(LabTests)}')
        print(f'Number of MicroBio   = {len(MicroBio)}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return new_Diagnosis, new_Medication, new_Procedures, new_LabTest, new_MicroBio



    def selecting_top_labs(self):
        self.update_statistics()
        node_degrees = {n: self.HG.degree(n) for n in self.Nodes if n[0] == 'L'}
        top_nodes = dict(sorted(node_degrees.items(), key=lambda item: item[1], reverse=True)[:480])
        labs_to_delete = [n for n in node_degrees if n not in top_nodes]
        self.HG.remove_nodes_from(labs_to_delete)
            

    def getDict2(self, df, id1, id2, c1, c2):
        # Create a copy of the relevant columns
        new_df = df[[id1, id2]].copy()
        
        # Drop rows with NaN values in either id1 or id2
        new_df = new_df.dropna(subset=[id1, id2])
        
        # Explicitly cast columns to string to avoid dtype compatibility issues
        new_df[id1] = new_df[id1].astype(str)
        new_df[id2] = new_df[id2].astype(str)
        
        # Add the prefixes to each column after ensuring there are no NaNs
        new_df.loc[:, id1] = c1 + '_' + new_df[id1]
        new_df.loc[:, id2] = c2 + '_' + new_df[id2]
        
        # Remove duplicate rows
        new_df = new_df.drop_duplicates()
        
        return new_df

    def getEdges(self, data, id1, id2):
        # Check if data is a DataFrame and extract edges accordingly
        if isinstance(data, pd.DataFrame):
            # Extract edges from the DataFrame
            EdgesList = list(data[[id1, id2]].itertuples(index=False, name=None))
        else:
            # Assuming data is a list of dictionaries
            EdgesList = [(d[id1], d[id2]) for d in data]
        
        return EdgesList


def G_statistics(G):
    Nodes = list(G.nodes())

    Patients =    [v for v in Nodes if v[0]=='C']
    Visits =      [v for v in Nodes if v[0]=='V']
    Medications = [v for v in Nodes if v[0]=='M']
    Diagnosis  =  [v for v in Nodes if v[0]=='D']
    Procedures =  [v for v in Nodes if v[0]=='P']
    Labs       =  [v for v in Nodes if v[0]=='L']
    MicroBio   =  [v for v in Nodes if v[0]=='B']
    

    print(f'number of patients = {len(Patients)}')
    print(f'number of visits = {len(Visits)}')
    print(f'number of Medication = {len(Medications)}')
    print(f'number of Diagnoses = {len(Diagnosis)}')
    print(f'number of Procedures = {len(Procedures)}')
    print(f'number of Labs = {len(Labs)}')
    print(f'number of MicoBio = {len(MicroBio)}')
    
    print(f'number of Edges = {G.number_of_edges()}')
    
    print('------------------------------------------\n')

def remove_patients_and_linked_visits(nodes, HG):
    """
    Remove patients and their linked visits from the graph HG.
    
    Parameters:
    nodes (list): List of patient nodes to be removed.
    HG (networkx.Graph): The heterogeneous graph.
    
    Returns:
    networkx.Graph: The modified graph with patients and their visits removed.
    """
    print(f'Number of PATIENTS to remove: {len(nodes)}')
    
    # Using a set to store nodes to avoid duplicates
    nodes_to_remove = set(nodes)
    
    # Find all visit nodes connected to the patient nodes
    for patient in nodes:
        visit_neighbors = {v for v in HG.neighbors(patient) if v[0] == 'V'}
        nodes_to_remove.update(visit_neighbors)
    
    print(f'Number of nodes to remove: {len(nodes_to_remove)}')

    # Removing nodes from the graph in place to avoid deepcopy
    HG.remove_nodes_from(nodes_to_remove)
    
    return HG

