import os
import pandas as pd

from data.pyg_dataset import PyGDataset
from data.data_split import ProteinSplit

# a first CED (PyGDataset or E3FPDataset) must be created to get the mol_ids
ced = PyGDataset()
rmsd_df = ced.mol_id_df.set_index('mol_id')

print('Computing ARMSD to all')
bio_rmsds = ced.fetch_bioactive_rmsds()
series_name = f'all'
rmsd_df = pd.concat([rmsd_df, pd.Series(bio_rmsds, name=series_name)], axis=1)

split_name = 'protein'
for split_i in range(5) :
    series_name = f'{split_name}_{split_i}'
    all_bio_rmsds = {}
    # we might find the same ligands in more than 1 dataset, but the proteins will be different and so will be the ARMSD
    for subset_name in ['train', 'val', 'test'] :
        data_split = ProteinSplit(split_name, split_i)
        smiles_list = data_split.get_smiles(subset_name)
        pdb_id_list = data_split.get_pdb_ids(subset_name)
        print(f'Computing ARMSD for {split_name}_{split_i}')
        bio_rmsds = ced.fetch_bioactive_rmsds(smiles_list, pdb_id_list)
        all_bio_rmsds.update(bio_rmsds)
        
    rmsd_df = pd.concat([rmsd_df, pd.Series(all_bio_rmsds, name=series_name)], axis=1)
    
rmsd_df_path = os.path.join(ced.root, 'rmsd_splits.csv')
rmsd_df.to_csv(rmsd_df_path)
