import os
import numpy as np
import pandas as pd
import torch

from abc import ABC, abstractmethod
from rdkit.Chem.rdchem import Mol
from typing import List, Sequence, Dict, Tuple
from torch.utils.data import Subset
from data.split.data_split import DataSplit
from tqdm import tqdm
from conf_ensemble import ConfEnsemble

class ConfEnsembleDataset(ABC):
    
    def __init__(self,
                 cel_name: str = 'pdb_conf_ensembles',
                 gen_cel_name: str = 'gen_conf_ensembles',
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/',
                 rmsd_df_filename: str = 'rmsd_splits.csv') -> None:
        self.cel_name = cel_name
        self.gen_cel_name = gen_cel_name
        self.root = root
        
        self.cel_dir = os.path.join(self.root, self.cel_name)
        self.gen_cel_dir = os.path.join(self.root, self.gen_cel_name)
        
        self.cel_df_path = os.path.join(self.cel_dir, 'ensemble_names.csv')
        self.cel_df = pd.read_csv(self.cel_df_path)
        
        self.rmsd_df_filename = rmsd_df_filename
        self.rmsd_df_path = os.path.join(self.root, self.rmsd_df_filename)
    
    
    def compute_mol_ids(self,
                        mol: Mol,
                        name: str) -> List[str]:
        mol_ids = []
        gen_i = 0
        
        confs = [conf for conf in mol.GetConformers()]
        for conf in confs :
            if conf.HasProp('PDB_ID'):
                pdb_id = conf.GetProp('PDB_ID')
                mol_id = f'{name}_{pdb_id}'
                mol_ids.append(mol_id)
            else:
                mol_id = f'{name}_Gen_{gen_i}'
                mol_ids.append(mol_id)
                gen_i = gen_i + 1
                
        return mol_ids
        
    
    def get_mol_id_subset_df(self,
                          smiles_list: Sequence = [],
                          pdb_id_list: Sequence = []) -> pd.DataFrame:
        mol_id_df = self.mol_id_df
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
                              pdb_id_list: Sequence = [],
                              rmsd_name: str = 'rmsds') :
        
        self.rmsd_name = rmsd_name
        self.rmsd_dir = os.path.join(self.root, self.rmsd_name)
        
        mol_id_subset_df = self.get_mol_id_subset_df(smiles_list, pdb_id_list)
        all_bioactive_rmsds = {}
        
        l = list(zip(self.cel_df['ensemble_name'], self.cel_df['filename']))
        for name, filename in tqdm(l):
            if name in mol_id_subset_df['ensemble_name'].values:
                try :
                    ce = self.get_merged_ce(filename, name)
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
            
        return all_bioactive_rmsds
    
    
    def get_bioactive_rmsds(self,
                            data_split: DataSplit) -> np.ndarray:
        rmsd_df = pd.read_csv(self.rmsd_df_path)
        split_type = data_split.split_type
        if split_type == 'protein':
            split_i = data_split.split_i
            col_name = f'{split_type}_{split_i}'
        else :
            col_name = 'all'
        rmsds = rmsd_df[col_name].values
        return rmsds
    
    
    @abstractmethod
    def add_bioactive_rmsds(self,
                            data_split: DataSplit) -> None:
        pass
    
    
    def get_merged_ce(self,
                      filename: str,
                      name: str) -> ConfEnsemble:
        ce_filepath = os.path.join(self.cel_dir, filename)
        conf_ensemble = ConfEnsemble.from_file(filepath=ce_filepath, 
                                                name=name)
        
        gen_conf_ensemble = self.get_generated_ce(filename, name)
        
        gen_conf_ensemble.add_mol(conf_ensemble.mol, standardize=False)
        return gen_conf_ensemble
    
    
    def get_generated_ce(self,
                         filename: str,
                         name: str,
                         output: str = 'conf_ensemble') -> ConfEnsemble:
        gen_ce_filepath = os.path.join(self.gen_cel_dir, filename)
        gen_conf_ensemble = ConfEnsemble.from_file(filepath=gen_ce_filepath, 
                                                    name=name,
                                                    output=output)
        return gen_conf_ensemble
    
    
    def get_split_subsets(self,
                          data_split: DataSplit
                          ) -> Dict[str, Tuple[Sequence[str], Subset]]:
        """
        Return a dict containing the train, validation and test sets based on 
        a data split
        
        :param data_split: Data split to be applied
        :type data_split: DataSplit
        :return: the splits in a dictionnary, with the train, val, and test keys
            to corresponding splits
        :rtype: Dict[str, Subset]
        
        """
        split_names = ['train', 'val', 'test']
        subsets = {}
        
        self.add_bioactive_rmsds(data_split) # Add the ARMSD adapted to the split
        
        # There is a 1:1 relationship between PDB id and smiles
        # While there is a 1:N relationship between smiles and PDB id
        # The second situation is able to select only certain bioactive conformation
        # for a molecule, while the first situation takes all pdb_ids for a smiles
        for split_name in split_names :
            subset = self.get_subset(data_split, split_name)
            try:
                mol_ids = self.mol_id_df['mol_id'][subset.indices]
            except:
                import pdb;pdb.set_trace()
            subsets[split_name] = (mol_ids, subset)
        return subsets
    
    
    def get_subset(self,
                   data_split: DataSplit,
                   subset_name: str) -> Subset:
        """
        Get a subset of the dataset, filtering using SMILES and/or PDB 
        identifiers
        
        :param data_split: Indicates how to split (giving allowed smiles/PDB ids)
        :type data_split: DataSplit
        :param split_name: Name of the subset from the split to get (train, val or test)
        :type split_name: str
        :return: filtered subset
        :rtype: Subset
        
        """
        # import pdb;pdb.set_trace()
        smiles_list = data_split.get_smiles(subset_name=subset_name)
        pdb_id_list = data_split.get_pdb_ids(subset_name=subset_name)
        mol_id_subset_df = self.get_mol_id_subset_df(smiles_list, pdb_id_list)
        indices = mol_id_subset_df.index
        subset = Subset(dataset=self, indices=indices)
        return subset
    