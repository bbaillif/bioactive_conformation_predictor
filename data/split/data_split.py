from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data import ConfEnsembleDataset
# the previous lines are to avoid cyclic dependencies
# between ConfEnsembleDataset and DataSplit

import os
import numpy as np
import pandas as pd

# from data import ConfEnsembleDataset
from conf_ensemble import ConfEnsembleLibrary
from tqdm import tqdm
from typing import List, Sequence
from abc import ABC, abstractmethod
from rdkit.Chem import Mol


class DataSplit(ABC) :
    
    def __init__(self,
                 split_type: str,
                 split_i: int,
                 cel_name: str = 'pdb_conf_ensembles',
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/',
                 splits_dirname: str = 'splits',
                 rmsd_name: str = 'rmsds') -> None:
        
        self.cel_name = cel_name
        self.root = root
        self.split_type = split_type
        self.split_i = split_i
        self.splits_dirname = splits_dirname
        self.rmsd_name = rmsd_name
        
        self.splits_dir_path = os.path.join(self.root, self.splits_dirname)
        self.split_dir_path = os.path.join(self.splits_dir_path,
                                           self.split_type,
                                           str(self.split_i))
        if not os.path.exists(self.split_dir_path):
            self.split_dataset()
        
        self.cel_dir = os.path.join(self.root, self.cel_name)
        
        cel_df_path = os.path.join(self.cel_dir, 'ensemble_names.csv')
        self.cel_df = pd.read_csv(cel_df_path)
        
        pdbbind_df_path = os.path.join(self.root, 'pdbbind_df.csv')
        self.pdbbind_df = pd.read_csv(pdbbind_df_path)
        
        self.rmsd_dir = os.path.join(self.root, self.rmsd_name)
        
    @abstractmethod
    def get_smiles(self,
                   subset_name: str) -> List[str]:
        pass
    
    
    @abstractmethod
    def get_pdb_ids(self,
                    subset_name: str) -> List[str]:
        pass
    
    @staticmethod
    @abstractmethod
    def split_dataset():
        pass
    
    
    def set_dataset(self,
                    dataset: ConfEnsembleDataset):
        self.dataset = dataset
    
    
    # Recursive function
    def get_bioactive_rmsds(self,
                            subset_name: str = 'train'):
        assert subset_name in ['train', 'val', 'test', 'all']
        rmsd_df_path = os.path.join(self.split_dir_path, 
                                    f'{subset_name}_rmsds.csv')
        if os.path.exists(rmsd_df_path):
            rmsd_df = pd.read_csv(rmsd_df_path)
        else:
            print('Compiling RMSD for given subset')
            rmsd_df = self.compute_bioactive_rmsds(subset_name)
            rmsd_df.to_csv(rmsd_df_path)
        return rmsd_df
        
        
    def compute_bioactive_rmsds(self,
                                subset_name: str):
        smiles_list = self.get_smiles(subset_name)
        pdb_id_list = self.get_pdb_ids(subset_name)
        print(f'Computing ARMSD for {self.split_type}_{self.split_i}')
        rmsd_df = self.fetch_bioactive_rmsds(smiles_list, 
                                               pdb_id_list)
        return rmsd_df
        
        
    def get_mol_id_subset_df(self,
                          smiles_list: Sequence = [],
                          pdb_id_list: Sequence = []) -> pd.DataFrame:
        mol_id_df = self.dataset.mol_id_df # defined in PyGDataset
        mol_id_df['ensemble_name'] = mol_id_df['mol_id'].apply(lambda s : s.split('_')[0])
        mol_id_df['pdb_id'] = mol_id_df['mol_id'].apply(lambda s : s.split('_')[1])
        mol_id_df = mol_id_df.merge(self.cel_df, on='ensemble_name') # adds smiles column
        if len(smiles_list) == 0 or len(pdb_id_list) == 0 :
            print('No filtering')
            mol_id_subset_df = mol_id_df
        else :
            print('Filtering according to input list')
            smiles_ok = mol_id_df['smiles'].isin(smiles_list)
            pdb_ok = mol_id_df['pdb_id'].isin(list(pdb_id_list) + ['Gen']) # Adding Gen to include generated conformations
            mol_id_subset_df = mol_id_df[smiles_ok & pdb_ok]
        return mol_id_subset_df
        
        
    def fetch_bioactive_rmsds(self,
                              smiles_list: Sequence = [],
                              pdb_id_list: Sequence = []) -> pd.DataFrame:
        
        mol_id_subset_df = self.get_mol_id_subset_df(smiles_list, pdb_id_list)
        all_bioactive_rmsds = {}
        
        l = list(zip(self.cel_df['ensemble_name'], self.cel_df['filename']))
        for name, filename in tqdm(l):
            if name in mol_id_subset_df['ensemble_name'].values:
                try :
                    ce = ConfEnsembleLibrary.get_merged_ce(filename, 
                                                           name)
                    mol = Mol(ce.mol)
                    
                    gen_i = 0
                    gen_mol_ids = []
                    bio_mol_ids = []
                    confs = [conf for conf in mol.GetConformers()]
                    for conf in confs :
                        if conf.HasProp('PDB_ID'):
                            pdb_id = conf.GetProp('PDB_ID')
                            mol_id = f'{name}_{pdb_id}'
                            bio_mol_ids.append(mol_id)
                        else:
                            mol_id = f'{name}_Gen_{gen_i}'
                            gen_mol_ids.append(mol_id)
                            gen_i = gen_i + 1
                    
                    file_prefix = filename.split('.')[0]
                    new_filename = f'{file_prefix}.npy'
                    filepath = os.path.join(self.rmsd_dir, new_filename)
                    rmsd_matrix = np.load(filepath)
                    
                    included_bio_idxs = [i
                                        for i, mol_id in enumerate(bio_mol_ids)
                                        if mol_id in mol_id_subset_df['mol_id'].values]
                
                
                    # we will take the minimum rmsd over bioactive conformations selected in our subset
                    rmsd_matrix_subset = rmsd_matrix[:, included_bio_idxs]
                    min_rmsds = rmsd_matrix_subset.min(1)
                    
                    # we assume that the generated conformations kept the same order
                    for i, mol_id in enumerate(gen_mol_ids) :
                        all_bioactive_rmsds[mol_id] = min_rmsds[i]
                        
                    for mol_id in bio_mol_ids :
                        all_bioactive_rmsds[mol_id] = 0
                
                except Exception as e:
                    print(f'Error with {name} {filename}')
                    print(e)
                    
        series = pd.Series(all_bioactive_rmsds, name='rmsd')
        rmsd_df = pd.DataFrame(series)
        rmsd_df.index.name = 'mol_id'
        return rmsd_df
    
    
    def get_bioschnet_checkpoint_path(self):
        checkpoint_dirname = os.path.join('lightning_logs',
                                          f'{self.split_type}_split_{self.split_i}',
                                          'checkpoints')
        checkpoint_filename = os.listdir(checkpoint_dirname)[0]
        checkpoint_filepath = os.path.join(checkpoint_dirname,
                                                  checkpoint_filename)
        return checkpoint_filepath