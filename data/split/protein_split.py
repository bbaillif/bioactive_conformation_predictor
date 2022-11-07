import os
import random

from abc import ABC
from .data_split import DataSplit

class ProteinSplit(DataSplit, ABC) :
    
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
    def get_pdb_ids(self, 
                    subset_name='all') :
        assert subset_name in ['train', 'val', 'test', 'all']
        if subset_name == 'all':
            all_pdb_ids = []
            for subset in ['train', 'val', 'test']:
                pdb_ids = self.get_pdb_ids(subset)
                all_pdb_ids.extend(pdb_ids)
        else:
            split_filename = f'{subset_name}_pdbs.txt'
            split_filepath = os.path.join(self.split_dir_path, 
                                          split_filename)
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