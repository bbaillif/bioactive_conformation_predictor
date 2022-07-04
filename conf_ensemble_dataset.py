from matplotlib.style import library
import torch
import os
import random
import numpy as np
import pandas as pd

from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.separate import separate
from rdkit import Chem
from tqdm import tqdm
from typing import Dict, List, Sequence
from molecule_featurizer import MoleculeFeaturizer
from conf_ensemble_library import ConfEnsembleLibrary
from data_split import DataSplit

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
random.seed(42)

class ConfEnsembleDataset(InMemoryDataset) :
    """
    Create a torch geometric dataset for each conformations in the default
    conf ensemble library (bio+gen)
    Args:
        :param root: Directory where library is stored
        :type root: str
        :param filter_out_bioactive: whether to add bioactive conformations
            in the dataset
        :type filter_out_bioactive: bool
    """
    
    def __init__(self, 
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/',
                 filter_out_bioactive: bool = False):
        
        self.root = root
        self.filter_out_bioactive = filter_out_bioactive
        
        self.smiles_df = pd.read_csv(os.path.join(self.root, 'smiles_df.csv'))
        
        is_included = self.smiles_df['included']
        self.included_data = self.smiles_df[is_included]
        self.included_smiles = self.included_data['smiles'].unique()
        
        self.molecule_featurizer = MoleculeFeaturizer()
        
        self.mol_id_df_filename = f'data_mol_ids.csv'
        self.mol_id_df_path = os.path.join(self.root, self.mol_id_df_filename)
        
        super().__init__(root=root) # calls the process functions
        assert os.path.exists(self.mol_id_df_path)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        return ['smiles_df.csv']
    
    @property
    def processed_file_names(self) -> List[str]:
        return [f'pdbbind_data.pt']
    
    
    def process(self):
        """
        Creates the dataset from the default library
        """
        
        cel = ConfEnsembleLibrary()
        cel.load_metadata()
        
        # Generate data
        all_data_list = []
        all_mol_ids = []
        for smiles in tqdm(self.included_smiles) :
            try :
                conf_ensemble = cel.load_ensemble_from_smiles(smiles,
                                                              library='merged')
                mol = conf_ensemble.mol
            
                mol_ids = []
                gen_i = 0
                confs = [conf for conf in mol.GetConformers()]
                for conf in confs :
                    if conf.HasProp('Generator') :
                        mol_id = f'{smiles}_Gen_{gen_i}'
                        mol_ids.append(mol_id)
                        gen_i = gen_i + 1
                    else :
                        pdb_id = conf.GetProp('PDB_ID')
                        mol_id = f'{smiles}_{pdb_id}'
                        mol_ids.append(mol_id)
                    
                data_list = self.molecule_featurizer.featurize_mol(mol, 
                                                                mol_ids=mol_ids)
                rmsds = self.molecule_featurizer.get_bioactive_rmsds(mol)
                for i, data in enumerate(data_list) :
                    data.rmsd = rmsds[i]
                    all_data_list.append(data)
                all_mol_ids.extend(mol_ids)
            except Exception as e :
                print(f'Error processing {smiles}')
                print(e)
                
        mol_id_df = pd.DataFrame({'mol_id' : all_mol_ids})
        mol_id_df.to_csv(self.mol_id_df_path)
                
        torch.save(self.collate(all_data_list), self.processed_paths[0])
            
            
    def get_smiles_for_pdb_ids(self,
                               pdb_ids: Sequence = []) -> np.ndarray:
        """
        Give the SMILES corresponding to the input PDB identifies using the 
        lookup table (smiles.csv)
        
        :param pdb_ids: sequence of PDB identifiers
        :type pdb_ids: Sequence
        :return: array of SMILES
        :rtype: np.ndarray
        
        """
        assert all([len(pdb_id) == 4 for pdb_id in pdb_ids])
        processed_data = self.smiles_df[self.smiles_df['id'].isin(pdb_ids)]
        processed_smiles = processed_data['smiles'].unique()
        return processed_smiles
            
            
    def get_pdb_ids_for_smiles(self,
                               smiles_list: Sequence = []) -> np.ndarray:
        """
        Give the PDB identitiers corresponding to the input SMILES using the 
        lookup table (smiles.csv)
        
        :param smiles_list: sequence of PDB identifiers
        :type smiles_list: Sequence
        :return: array of PDB identifiers
        :rtype: np.ndarray
        """
        processed_data = self.smiles_df[self.smiles_df['smiles'].isin(smiles_list)]
        processed_pdb_ids = processed_data['id'].unique()
        return processed_pdb_ids
    
    
    def get_subset(self,
                   smiles_list: Sequence = [],
                   pdb_id_list: Sequence = []) -> Subset:
        """
        Get a subset of the dataset, filtering using SMILES and/or PDB 
        identifiers
        
        :param smiles_list: List of allowed SMILES
        :type smiles_list: Sequence
        :param pdb_id_list: List of allowed PDB identifiers
        :type pdb_id_list: Sequence
        :return: filtered subset
        :rtype: Subset
        
        """
        mol_id_df = pd.read_csv(self.mol_id_df_path, index_col=0)
        mol_id_df['smiles'] = mol_id_df['mol_id'].apply(lambda s : s.split('_')[0])
        mol_id_df['pdb_id'] = mol_id_df['mol_id'].apply(lambda s : s.split('_')[1])
        smiles_ok = mol_id_df['smiles'].isin(smiles_list)
        pdb_ok = mol_id_df['pdb_id'].isin(list(pdb_id_list) + ['Gen']) # Adding Gen to include generated conformations
        indices = mol_id_df[smiles_ok & pdb_ok].index
        subset = Subset(dataset=self, indices=indices)
        
        return subset
    
    
    def get_splits(self,
                   data_split: DataSplit) -> Dict[str, Subset]:
        """
        Return a dict containing the train, validation and test sets based on 
        a data split
        
        :param data_split: Data split to be applied
        :type data_split: DataSplit
        :return: the splits in a dictionnary, with the train, val, and test keys
            to corresponding splits
        :rtype: Dict[str, Subset]
        
        """
        split_type = data_split.split_type
        split_names = ['train', 'val', 'test']
        subsets = {}
        
        # There is a 1:1 relationship between PDB id and smiles
        # While there is a 1:N relationship between smiles and PDB id
        # The second situation is able to select only certain bioactive conformation
        # for a molecule, while the first situation takes all pdb_ids for a smiles
        for split_name in split_names :
            if split_type in ['random', 'scaffold', 'no_split'] :
                processed_smiles = data_split.get_smiles(dataset=split_name)
                processed_pdb_ids = self.get_pdb_ids_for_smiles(processed_smiles)
            elif split_type in ['protein'] :
                processed_pdb_ids = data_split.get_pdb_ids(dataset=split_name)
                processed_smiles = self.get_smiles_for_pdb_ids(processed_pdb_ids)
            subset = self.get_subset(smiles_list=processed_smiles,
                                        pdb_id_list=processed_pdb_ids)
            subsets[split_name] = subset
        return subsets
    
if __name__ == '__main__':
    conf_ensemble_dataset = ConfEnsembleDataset()
    