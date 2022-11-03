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
        
        
        
class MoleculeSplit(DataSplit) :
    
    def __init__(self,
                 split_type: str='random',
                 split_i: int=0) -> None:
        assert split_type in ['random', 'scaffold']
        super().__init__(split_type=split_type,
                         split_i=split_i)
        self.split_dir = os.path.join(self.root,
                                      f'{split_type}_splits')
        
        
    # recursive function
    def get_smiles(self, 
                   subset_name='all') -> List[str]:
        assert subset_name in ['train', 'val', 'test', 'all']
        if subset_name == 'all' :
            all_smiles = []
            # iterate over all possible subsets
            for subset_name in ['train', 'val', 'test']:
                smiles = self.get_smiles(subset_name)
                all_smiles.extend(smiles)
        else :
            split_filename = f'{subset_name}_smiles_{self.split_i}.txt'
            split_filepath = os.path.join(self.split_dir, split_filename)
            with open(split_filepath, 'r') as f :
                smiles = [s.strip() for s in f.readlines()]
            all_smiles = smiles
        return all_smiles
    
    
    def get_pdb_ids(self, 
                    subset_name='all') :
        assert subset_name in ['train', 'val', 'test', 'all']
        smiles = self.get_smiles(subset_name)
        names = self.cel_df[self.cel_df['smiles'].isin(smiles)]['ensemble_name'].unique()
        pdb_ids = self.pdbbind_df[self.pdbbind_df['ligand_name'].isin(names)]['pdb_id'].unique()
        return pdb_ids

class ProteinSplit(DataSplit) :
    
    def __init__(self, 
                 split_type: str = 'protein', 
                 split_i: int = 0) -> None:
        assert split_type in ['protein']
        super().__init__(split_type=split_type,
                         split_i=split_i)
        self.split_dir = os.path.join(self.root,
                                      f'{split_type}_splits')
        
    # recursive function
    def get_pdb_ids(self, 
                    subset_name='all') :
        assert subset_name in ['train', 'val', 'test', 'all']
        if subset_name == 'all':
            all_pdb_ids = []
            for subset in ['train', 'val', 'test']:
                pdb_ids = self.get_pdb_ids(subset)
                all_pdb_ids.extend(pdb_ids)
        else:
            split_filename = f'{subset_name}_pdb_{self.split_i}.txt'
            split_filepath = os.path.join(self.split_dir, split_filename)
            with open(split_filepath, 'r') as f :
                pdb_ids = [s.strip() for s in f.readlines()]
            all_pdb_ids = pdb_ids
        return all_pdb_ids
    
    
    def get_smiles(self, 
                   subset_name='all') :
        assert subset_name in ['train', 'val', 'test', 'all']
        pdb_ids = self.get_pdb_ids(subset_name)
        names = self.pdbbind_df[self.pdbbind_df['pdb_id'].isin(pdb_ids)]['ligand_name'].unique()
        smiles = self.cel_df[self.cel_df['ensemble_name'].isin(names)]['smiles'].unique()
        return smiles
    

class NoSplit(DataSplit) :
    
    def __init__(self) -> None:
        self.split_type = 'no_split'
        self.split_i = 0
        super().__init__(split_type=self.split_type, 
                         split_i=self.split_i)
        
    def get_smiles(self,
                   subset_name=None) :
        return self.cel_df['smiles'].unique()