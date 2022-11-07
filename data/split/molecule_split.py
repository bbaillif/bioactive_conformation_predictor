import os
from .data_split import DataSplit
from typing import List
from abc import ABC

class MoleculeSplit(DataSplit, ABC) :
    
    def __init__(self, 
                 split_type: str, 
                 split_i: int, 
                 cel_name: str = 'pdb_conf_ensembles', 
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/', 
                 splits_dirname: str = 'splits', 
                 rmsd_name: str = 'rmsds') -> None:
        super().__init__(split_type, 
                         split_i, 
                         cel_name, 
                         root, 
                         splits_dirname, 
                         rmsd_name)
        
        
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
            split_filename = f'{subset_name}_smiles.txt'
            split_filepath = os.path.join(self.split_dir_path, 
                                          split_filename)
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