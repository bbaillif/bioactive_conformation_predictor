from rdkit import Chem
from ccdc_rdkit_connector import CcdcRdkitConnector

import os
import pandas as pd
import argparse

from conf_ensemble import ConfEnsembleLibrary

parser = argparse.ArgumentParser(description='Select data directory')
parser.add_argument('--data_dir', 
                    type=str, 
                    default='data/',
                    help='directory where data are stored')
args = parser.parse_args()
data_dir_path = args.data_dir

all_CEL = ConfEnsembleLibrary()
all_CEL.load_metadata()
    
ccdc_rdkit_connector = CcdcRdkitConnector()

smiled_df_path = os.path.join(data_dir_path, 'smiles_df.csv')
smiles_df = pd.read_csv(smiled_df_path, index_col=0)
included_smiles = smiles_df[smiles_df['included']]['smiles'].unique()

all_CEL.generate_conf_pool(included_smiles=included_smiles)