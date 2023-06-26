import os
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from rdkit.Chem import Mol
from typing import List, Sequence, Dict, Tuple
from torch.utils.data import Subset
from data.split import DataSplit
from tqdm import tqdm
from conf_ensemble import ConfEnsembleLibrary

class ConfEnsembleDataset(ABC):
    """
    Base class to handle dataset of conformations. Uses ConfEnsembleLibrary
    to handle different conformers for the same molecules.
    Use case is having dataset of bioactive+generated conformations,
    where each type of conformation is in a different library
    
    :param cel_name: Name of the bioactive conformations library
    :type cel_name: str
    :param gen_cel_name: Name of the generated conformations library
    :type gen_cel_name: str
    :param root: Data directory:
    :type root: str
    
    """
    
    def __init__(self,
                 cel_name: str = 'pdb_conf_ensembles',
                 gen_cel_name: str = 'gen_conf_ensembles',
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/') -> None:
        self.cel_name = cel_name
        self.gen_cel_name = gen_cel_name
        self.root = root
        
        self.cel_dir = os.path.join(self.root, self.cel_name)
        self.gen_cel_dir = os.path.join(self.root, self.gen_cel_name)
        
        self.cel_df_path = os.path.join(self.cel_dir, 'ensemble_names.csv')
    
    
    def compute_mol_ids(self,
                        mol: Mol,
                        name: str) -> List[str]:
        """
        Compute the mol_ids (one per conformation) based on molecule name and
        whether it is bioactive (PDB ID) or generated (number)
        
        :param mol: input RDKit molecule
        :type mol: Mol
        :param name: name of the input molecule
        :type name: str
        """
        mol_ids = []
        gen_i = 0
        
        confs = [conf for conf in mol.GetConformers()]
        for conf in confs :
            if conf.HasProp('PDB_ID'):
                pdb_id = conf.GetProp('PDB_ID')
                mol_id = f'{name}__{pdb_id}'
                mol_ids.append(mol_id)
            else:
                mol_id = f'{name}__Gen__{gen_i}'
                mol_ids.append(mol_id)
                gen_i = gen_i + 1
                
        return mol_ids
        
    
    def get_mol_id_subset_df(self,
                          smiles_list: List = [],
                          pdb_id_list: List = []) -> pd.DataFrame:
        """
        Based on input smiles and pdb_ids, create a subset of the mol_id dataframe
        Special case is when one the input list is empty, then no filtering is done
        and the full dataset is returned
        
        :param smiles_list: List of molecule SMILES to keep
        :type smiles_list: List
        :param pdb_id_list: List of PDB IDs to keep
        :type pdb_id_list: List
        
        """
        self.cel_df = pd.read_csv(self.cel_df_path)
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
        """
        Retrieve the ARMSD to bioactive conformation for each molecule id
        based on the input lists
        
        :param smiles_list: List of molecule SMILES to keep
        :type smiles_list: List
        :param pdb_id_list: List of PDB IDs to keep
        :type pdb_id_list: List
        :param rmsd_name: Name of the directory containing ARMSD matrices 
            (generated * bioactive) for each molecule in CEL
        :type rmsd_name: str
        
        """
        self.cel_df = pd.read_csv(self.cel_df_path)
        rmsd_dir = os.path.join(self.root, rmsd_name)
        
        mol_id_subset_df = self.get_mol_id_subset_df(smiles_list, pdb_id_list)
        all_bioactive_rmsds = {}
        
        l = list(zip(self.cel_df['ensemble_name'], self.cel_df['filename']))
        for name, filename in tqdm(l):
            if name in mol_id_subset_df['ensemble_name'].values:
                try :
                    ce = ConfEnsembleLibrary.get_merged_ce(filename, 
                                                           name,
                                                           root=self.root,
                                                           cel_name1=self.cel_name,
                                                           cel_name2=self.gen_cel_name)
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
                    filepath = os.path.join(rmsd_dir, new_filename)
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
    
    
    @abstractmethod
    def add_bioactive_rmsds(self,
                            data_split: DataSplit,
                            subset_name: str) -> None:
        """
        Setup the conformations targets (ARMSD) based on the data_split and
        given subset (train, val or test)
        
        :param data_split: DataSplit object handling how the data is split
        :type data_split: DataSplit
        :param subset_name: 'train', 'val' or 'test'
        :type subset_name: str
        
        """
        pass
    
    
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
        mol_id_df = self.mol_id_df
        assert self.cel_df_path == data_split.cel_df_path, \
            'The data_split and dataset must have the same CEL directory'
        mol_id_subset_df = data_split.get_mol_id_subset_df(mol_id_df,
                                                           smiles_list, 
                                                           pdb_id_list)
        indices = mol_id_subset_df.index
        subset = Subset(dataset=self, indices=indices)
        return subset
    