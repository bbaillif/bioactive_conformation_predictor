import os
from data_split import DataSplit

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