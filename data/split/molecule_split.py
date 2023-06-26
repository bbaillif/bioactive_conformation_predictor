import os
from .data_split import DataSplit
from typing import List
from abc import ABC
from params import (DATA_DIRPATH,
                    BIO_CONF_DIRNAME,
                    GEN_CONF_DIRNAME,
                    RMSD_DIRNAME,
                    SPLITS_DIRNAME)

class MoleculeSplit(DataSplit, ABC) :
    """Class for the molecule split: the complexes are split based on their 
    unique ligands between the train, val and test subsets

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
        super().__init__(split_type, 
                         split_i, 
                         cel_name, 
                         gen_cel_name,
                         root, 
                         splits_dirname, 
                         rmsd_name)
        
        
    # recursive function
    def get_smiles(self, 
                   subset_name='all') -> List[str]:
        """ Get the smiles for all molecules in a subset (train, val or test)

        :param subset_name: train, val, test, or all, defaults to 'all'
        :type subset_name: str, optional
        :return: List of smiles
        :rtype: List[str]
        """
        assert subset_name in ['train', 'val', 'test', 'all']
        if subset_name == 'all' :
            all_smiles = []
            # iterate over all possible subsets
            for subset_name in ['train', 'val', 'test']:
                smiles = self.get_smiles(subset_name)
                all_smiles.extend(smiles)
        else :
            split_filename = f'{subset_name}_smiles.txt'
            split_filepath = os.path.join(self.split_dir_path, 
                                          split_filename)
            with open(split_filepath, 'r') as f :
                smiles = [s.strip() for s in f.readlines()]
            all_smiles = smiles
        return all_smiles
    
    
    def get_pdb_ids(self, 
                    subset_name='all') -> List[str]:
        """Get the pdb_ids for all ligands/complexes in the subset (train, 
        val or test)

        :param subset_name: train, val, test, or all, defaults to 'all'
        :type subset_name: str, optional
        :return: List of PDB ids
        :rtype: List[str]
        """
        assert subset_name in ['train', 'val', 'test', 'all']
        smiles = self.get_smiles(subset_name)
        names = self.cel_df[self.cel_df['smiles'].isin(smiles)]['ensemble_name'].unique()
        pdb_ids = self.pdb_df[self.pdb_df['ligand_name'].isin(names)]['pdb_id'].unique()
        return pdb_ids