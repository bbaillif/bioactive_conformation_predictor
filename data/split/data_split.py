import os
import pandas as pd

from typing import List
from abc import ABC, abstractmethod

class DataSplit(ABC) :
    
    def __init__(self,
                 cel_name: str = 'pdb_conf_ensembles',
                 root: str='/home/bb596/hdd/pdbbind_bioactive/data/',
                 split_type: str='random',
                 split_i: int=0) -> None:
        
        self.cel_name = cel_name
        self.root = root
        self.split_type = split_type
        self.split_i = split_i
        
        self.cel_dir = os.path.join(self.root, self.cel_name)
        
        cel_df_path = os.path.join(self.cel_dir, 'ensemble_names.csv')
        self.cel_df = pd.read_csv(cel_df_path)
        
        pdbbind_df_path = os.path.join(self.root, 'pdbbind_df.csv')
        self.pdbbind_df = pd.read_csv(pdbbind_df_path)
        
    @abstractmethod
    def get_smiles(self,
                   subset_name: str) -> List[str]:
        pass
    
    
    @abstractmethod
    def get_pdb_ids(self,
                    subset_name: str) -> List[str]:
        pass