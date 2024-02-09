import os
import numpy as np

from .protein_split import ProteinSplit
from sklearn.model_selection import train_test_split
from data.pdbbind import PDBbindMetadataProcessor
from data.chembl_connector import ChEMBLConnector

class KinasesTKSplit(ProteinSplit):
    
    def __init__(self, 
                 split_type: str = 'kinases_tk', 
                 split_i: int = 0) -> None:
        super().__init__(split_type, split_i)
        
    def split_dataset(self):
        pmp = PDBbindMetadataProcessor(root='/home/bb596/hdd/PDBbind/',
                                       remove_unknown_ligand_name=True,
                                       remove_unknown_uniprot=True)
        pdbbind_table = pmp.get_master_dataframe()
        
        # Select Kinases
        cc = ChEMBLConnector()
        chembl_target_df = cc.get_target_table(level=4)
        pdbbind_table = pdbbind_table.merge(chembl_target_df, left_on='Uniprot ID', right_on='accession')
        pdbbind_table = pdbbind_table[pdbbind_table['protein_class_desc'].str.contains('protein kinase')]
        
        l_class = ['enzyme  kinase  protein kinase  tk']
        condition = pdbbind_table['level4'].isin(l_class)
        class_pdbs = pdbbind_table[condition]['PDB code'].values
        other_pdbs = pdbbind_table[~condition]['PDB code'].values

        kinase_splits_dir_path = os.path.join(self.splits_dir_path, 
                                              self.split_type)
        if not os.path.exists(kinase_splits_dir_path) :
            os.mkdir(kinase_splits_dir_path)

        seed = 42
        for i in range(5) :
            
            current_split_dir_path = os.path.join(kinase_splits_dir_path, str(i))
            if not os.path.exists(current_split_dir_path):
                os.mkdir(current_split_dir_path)
            
            train_pdbs, test_pdbs = train_test_split(class_pdbs, train_size=0.8, random_state=seed)
            val_pdbs, test_pdbs = train_test_split(test_pdbs, train_size=0.5, random_state=seed)
            test_pdbs = np.hstack([other_pdbs, test_pdbs])
            
            with open(os.path.join(current_split_dir_path, f'train_pdbs.txt'), 'w') as f :
                for pdb in train_pdbs :
                    f.write(pdb)
                    f.write('\n')
                
            with open(os.path.join(current_split_dir_path, f'val_pdbs.txt'), 'w') as f :
                for pdb in val_pdbs :
                    f.write(pdb)
                    f.write('\n')
                
            with open(os.path.join(current_split_dir_path, f'test_pdbs.txt'), 'w') as f :
                for pdb in test_pdbs :
                    f.write(pdb)
                    f.write('\n')
            
            seed = seed + 1