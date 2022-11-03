import os
from data_split import DataSplit
from typing import List

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