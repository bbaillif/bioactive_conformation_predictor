import os
import pandas as pd

class DataSplit() :
    
    def __init__(self,
                 root: str='/home/bb596/hdd/pdbbind_bioactive/data/',
                 split_type: str='random',
                 split_i: int=0) -> None:
        self.root = root
        self.split_type = split_type
        self.split_i = split_i
        
        self.smiles_df = pd.read_csv(os.path.join(self.root, 'smiles_df.csv'))
        
        
class MoleculeSplit(DataSplit) :
    
    def __init__(self,
                 split_type: str='random',
                 split_i: int=0) -> None:
        assert split_type in ['random', 'scaffold']
        super().__init__(split_type=split_type,
                         split_i=split_i)
        self.split_dir = os.path.join(self.root,
                                      f'ligand_{split_type}_splits')
        
    def get_smiles(self, dataset='train') :
        assert dataset in ['train', 'val', 'test']
        split_filename = f'{dataset}_smiles_{self.split_type}_split_{self.split_i}.txt'
        split_filepath = os.path.join(self.split_dir, split_filename)
        with open(split_filepath, 'r') as f :
            smiles = [s.strip() for s in f.readlines()]
        return smiles
    

class ProteinSplit(DataSplit) :
    
    def __init__(self, 
                 split_type: str = 'protein', 
                 split_i: int = 0) -> None:
        assert split_type in ['protein']
        super().__init__(split_type=split_type,
                         split_i=split_i)
        self.split_dir = os.path.join(self.root,
                                      f'{split_type}_similarity_splits')
        
    def get_pdb_ids(self, 
                    dataset='train') :
        assert dataset in ['train', 'val', 'test']
        split_filename = f'{dataset}_pdb_{self.split_type}_similarity_split_{self.split_i}.txt'
        split_filepath = os.path.join(self.split_dir, split_filename)
        with open(split_filepath, 'r') as f :
            pdb_ids = [s.strip() for s in f.readlines()]
        return pdb_ids
    
    def get_smiles(self, 
                   dataset='train') :
        assert dataset in ['train', 'val', 'test']
        pdb_ids = self.get_pdb_ids(dataset)
        smiles = self.smiles_df[self.smiles_df['id'].isin(pdb_ids)]['smiles'].unique()
        return smiles
    

class NoSplit(DataSplit) :
    
    def __init__(self) -> None:
        self.split_type = 'no_split'
        self.split_i = 0
        super().__init__(split_type=self.split_type, 
                         split_i=self.split_i)
        
    def get_smiles(self,
                   dataset=None) :
        return self.smiles_df[self.smiles_df['included']]['smiles'].unique()[:100]