# from __future__ import annotations
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from data import ConfEnsembleDataset
# the previous lines are to avoid cyclic dependencies
# between ConfEnsembleDataset and DataSplit

import os
import numpy as np
import pandas as pd

# from data import ConfEnsembleDataset
from bioconfpred.conf_ensemble import ConfEnsembleLibrary
from tqdm import tqdm
from typing import List, Sequence
from abc import ABC, abstractmethod
from rdkit.Chem import Mol
# from model import SchNetModel, DimeNetModel, ComENetModel
from bioconfpred.params import (DATA_DIRPATH,
                    BIO_CONF_DIRNAME,
                    GEN_CONF_DIRNAME,
                    RMSD_DIRNAME,
                    SPLITS_DIRNAME)


class DataSplit(ABC) :
    """Base class for data splits (definition of which ligand/bioactive
    conformation is in the train, val or test subset)

    :param split_type: Name of the split type
    :type split_type: str
    :param split_i: Number of the split (iteration)
    :type split_i: int
    :param cel_name: Name of the bioactive conformation ensemble library, 
        defaults to 'pdb_conf_ensembles'
    :type cel_name: str, optional
    :param gen_cel_name: Name of the generated conformer ensemble library, 
        defaults to 'gen_conf_ensembles'
    :type gen_cel_name: str, optional
    :param root: Data directory
    :type root: str, optional
    :param splits_dirname: Name of the directory where the splits information
        are stored, defaults to 'splits'
    :type splits_dirname: str, optional
    :param rmsd_name: Name of the directory storing the precomputed RMSD, 
        defaults to 'rmsds'
    :type rmsd_name: str, optional
    """
    
    def __init__(self,
                 split_type: str,
                 split_i: int,
                 cel_name: str = BIO_CONF_DIRNAME,
                 gen_cel_name: str = GEN_CONF_DIRNAME,
                 root: str = DATA_DIRPATH,
                 splits_dirname: str = SPLITS_DIRNAME,
                 rmsd_name: str = RMSD_DIRNAME) -> None:
        
        
        self.cel_name = cel_name
        self.gen_cel_name = gen_cel_name
        self.root = root
        self.split_type = split_type
        self.split_i = split_i
        self.splits_dirname = splits_dirname
        self.rmsd_name = rmsd_name
        
        self.cel_dir = os.path.join(self.root, self.cel_name)
        
        self.cel_df_path = os.path.join(self.cel_dir, 'ensemble_names.csv')
        self.cel_df = pd.read_csv(self.cel_df_path)
        
        pdb_df_path = os.path.join(self.cel_dir, 'pdb_df.csv')
        self.pdb_df = pd.read_csv(pdb_df_path)
        
        self.rmsd_dir = os.path.join(self.root, self.rmsd_name)
        
        self.splits_dir_path = os.path.join(self.root, self.splits_dirname)
        if not os.path.exists(self.splits_dir_path):
            os.mkdir(self.splits_dir_path)
        
        self.split_dir_path = os.path.join(self.splits_dir_path,
                                           self.split_type,
                                           str(self.split_i))
        if not os.path.exists(self.split_dir_path):
            self.split_dataset()
        
    @abstractmethod
    def get_smiles(self,
                   subset_name: str) -> List[str]:
        """Get the smiles of molecules in a subset (train, val or test)

        :param subset_name: train, val or test
        :type subset_name: str
        :return: list of smiles
        :rtype: List[str]
        """
        pass
    
    
    @abstractmethod
    def get_pdb_ids(self,
                    subset_name: str) -> List[str]:
        """Get the PDB ids of molecules in a subset (train, val or test)

        :param subset_name: train, val or test
        :type subset_name: str
        :return: list of PDB ids
        :rtype: List[str]
        """
        pass
    
    @staticmethod
    @abstractmethod
    def split_dataset():
        """Performs the split
        """
        pass
    
    
    def check_fractions(self,
                        train_fraction: float,
                        val_fraction: float) -> None:
        assert train_fraction > 0 and train_fraction < 1, \
            'Train fraction should be between 0 and 1'
        assert val_fraction > 0 and val_fraction < 1, \
            'Validation fraction should be between 0 and 1'
        assert (train_fraction + val_fraction) < 1, \
            'Sum of train and val fraction should be lower than 1 to allow a test subset'

    
    def get_bioactive_rmsds(self,
                            mol_id_df: pd.DataFrame,
                            subset_name: str = 'train') -> pd.DataFrame:
        """Get the compiled bioactive rmsds for the given mol_ids and subset

        :param mol_id_df: Dataframe storing mol_ids to include
        :type mol_id_df: pd.DataFrame
        :param subset_name: train, val or test, defaults to 'train'
        :type subset_name: str, optional
        :return: Dataframe storing the rmsds for each mol_id
        :rtype: pd.DataFrame
        """
        
        assert subset_name in ['train', 'val', 'test', 'all']
        rmsd_df_path = os.path.join(self.split_dir_path, 
                                    f'{subset_name}_rmsds.csv')
        if os.path.exists(rmsd_df_path):
            rmsd_df = pd.read_csv(rmsd_df_path)
        else:
            print('Compiling RMSD for given subset')
            rmsd_df = self.compile_bioactive_rmsds(mol_id_df, subset_name)
            rmsd_df.to_csv(rmsd_df_path)
        return rmsd_df
        
        
    def compile_bioactive_rmsds(self,
                                mol_id_df: pd.DataFrame,
                                subset_name: str = 'train') -> pd.DataFrame:
        """Compile the bioactive rmsds

        :param mol_id_df: Dataframe storing mol_ids to include
        :type mol_id_df: pd.DataFrame
        :param subset_name: train, val or test, defaults to 'train'
        :type subset_name: str, optional
        :return: Dataframe storing the rmsds for each mol_id
        :rtype: pd.DataFrame
        """
        smiles_list = self.get_smiles(subset_name)
        pdb_id_list = self.get_pdb_ids(subset_name)
        print(f'Computing ARMSD for {self.split_type}_{self.split_i}')
        rmsd_df = self.fetch_bioactive_rmsds(mol_id_df,
                                             smiles_list, 
                                             pdb_id_list)
        return rmsd_df
        
        
    def get_mol_id_subset_df(self,
                             mol_id_df: pd.DataFrame,
                             smiles_list: Sequence = [],
                             pdb_id_list: Sequence = []) -> pd.DataFrame:
        """Get a subset of mol_ids based on an input list of smiles and pdb_ids.
        If the list of input smiles and list of input pdb_ids are empty, no 
        filtering is applied.

        :param mol_id_df: Initial DataFrame of mol_ids
        :type mol_id_df: pd.DataFrame
        :param smiles_list: List of input smiles, defaults to []
        :type smiles_list: Sequence, optional
        :param pdb_id_list: List of input pdb_ids, defaults to []
        :type pdb_id_list: Sequence, optional
        :return: Dataframe of filtered mol_ids
        :rtype: pd.DataFrame
        """
        
        def get_first_split_element(s: str): return s.split('__')[0]
        def get_second_split_element(s: str): return s.split('__')[1]
        
        mol_id_df['ensemble_name'] = mol_id_df['mol_id'].apply(get_first_split_element)
        mol_id_df['origin'] = mol_id_df['mol_id'].apply(get_second_split_element)
        
        mol_id_df = mol_id_df.merge(self.cel_df, on='ensemble_name') # adds smiles column
        if len(smiles_list) == 0 or len(pdb_id_list) == 0 :
            print('No filtering')
            mol_id_subset_df = mol_id_df
        else :
            print('Filtering according to input lists')
            smiles_ok = mol_id_df['smiles'].isin(smiles_list)
            pdb_ok = mol_id_df['origin'].isin(list(pdb_id_list) + ['Gen']) # Adding Gen to include generated conformations
            mol_id_subset_df = mol_id_df[smiles_ok & pdb_ok]
        return mol_id_subset_df
        
        
    def fetch_bioactive_rmsds(self,
                              mol_id_df: pd.DataFrame,
                              smiles_list: Sequence = [],
                              pdb_id_list: Sequence = []) -> pd.DataFrame:
        """Fetch the precomputed RMSDs and compile them in a Dataframe

        :param mol_id_df: Dataframe of input mol_ids
        :type mol_id_df: pd.DataFrame
        :param smiles_list: List of input smiles, defaults to []
        :type smiles_list: Sequence, optional
        :param pdb_id_list: List of input pdb_ids, defaults to []
        :type pdb_id_list: Sequence, optional
        :return: Dataframe of mol_ids and corresponding RMSD
        :rtype: pd.DataFrame
        """
        
        mol_id_subset_df = self.get_mol_id_subset_df(mol_id_df,
                                                     smiles_list, 
                                                     pdb_id_list)
        all_bioactive_rmsds = {}
        
        l = list(zip(self.cel_df['ensemble_name'], self.cel_df['filename']))
        for name, filename in tqdm(l):
            if name in mol_id_subset_df['ensemble_name'].values:
                try :
                    ce = ConfEnsembleLibrary.get_merged_ce(filename, 
                                                           name,
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
                            mol_id = f'{name}__{pdb_id}'
                            bio_mol_ids.append(mol_id)
                        else:
                            mol_id = f'{name}__Gen__{gen_i}'
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
                    import pdb;pdb.set_trace()
                    print(e)
                    
        series = pd.Series(all_bioactive_rmsds, name='rmsd')
        rmsd_df = pd.DataFrame(series)
        rmsd_df.index.name = 'mol_id'
        return rmsd_df
    
    
    # def get_model_checkpoint(self,
    #                          model_name: str = 'schnet',
    #                          log_dir: str = 'lightning_logs'):
        
    #     experiment_name = f'{self.split_type}_split_{self.split_i}'
    #     # TODO: modify model_name and experiment_name handling
    #     if model_name == 'comenet':
    #         experiment_name = experiment_name + '_comenet'
    #     elif model_name == 'dimenet':
    #         experiment_name = experiment_name + '_dimenet'
    #     elif model_name == 'schnet':
    #         experiment_name = experiment_name + '_schnet'
    #     else:
    #         raise Exception('Unknown model name')
        
    #     checkpoint_dirname = os.path.join(self.root, 
    #                                       log_dir,
    #                                       experiment_name,
    #                                       'checkpoints')
    #     checkpoint_filename = os.listdir(checkpoint_dirname)[0]
    #     checkpoint_path = os.path.join(checkpoint_dirname,
    #                                               checkpoint_filename)
        
    #     if model_name == 'comenet':
    #         config = {"lr": 1e-5,
    #             'batch_size': 256,
    #             'data_split': self}
    #         model = ComENetModel.load_from_checkpoint(checkpoint_path=checkpoint_path,
    #                                                   config=config)
    #     elif model_name == 'dimenet':
    #         config = {'hidden_channels': 128,
    #                 'out_channels': 1,
    #                 'num_blocks': 4,
    #                 'int_emb_size': 64,
    #                 'basis_emb_size': 8,
    #                 'out_emb_channels': 256,
    #                 'num_spherical': 7,
    #                 'num_radial':6 ,
    #               "lr":1e-4,
    #               'batch_size': 256,
    #               'data_split': self}
    #         model = DimeNetModel.load_from_checkpoint(checkpoint_path=checkpoint_path, 
    #                                                   config=config)
    #     elif model_name == 'schnet':
    #         config = {"num_interactions": 6,
    #                 "cutoff": 10,
    #                 "lr": 1e-5,
    #                 'batch_size': 256,
    #                 'data_split': self}
    #         model = SchNetModel.load_from_checkpoint(checkpoint_path=checkpoint_path, 
    #                                                  config=config)
        
    #     return model
    
    
    # def get_bioschnet_checkpoint_path(self):
    #     checkpoint_dirname = os.path.join('../hdd/pdbbind_bioactive/data/lightning_logs',
    #                                       f'{self.split_type}_split_{self.split_i}',
    #                                       'checkpoints')
    #     checkpoint_filename = os.listdir(checkpoint_dirname)[0]
    #     checkpoint_filepath = os.path.join(checkpoint_dirname,
    #                                               checkpoint_filename)
    #     return checkpoint_filepath