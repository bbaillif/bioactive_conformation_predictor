import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit.Chem.rdMolAlign import GetBestRMS
from typing import List, Tuple
from rdkit.Chem import Mol
from multiprocessing import Pool
from bioconfpred.conf_ensemble import ConfEnsembleLibrary
from bioconfpred.data.utils import MolConverter
from bioconfpred.params import (DATA_DIRPATH,
                    BIO_CONF_DIRNAME,
                    GEN_CONF_DIRNAME,
                    RMSD_DIRNAME)

try:
    from ccdc.descriptors import MolecularDescriptors
except:
    print('CSD Python API not installed')

class RMSDCalculator() :
    """Class to generate RMSD between bioactive conformations and generated
    conformers

    :param rmsd_name: Name of the directory to store the RMSDS, 
        defaults to 'rmsds'
    :type rmsd_name: str, optional
    :param cel_name1: Name of the generated conformer ensemble library, 
        defaults to 'gen_conf_ensembles/'
    :type cel_name1: str, optional
    :param cel_name2: Name of the bioactive conformation ensemble library, 
        defaults to 'pdb_conf_ensembles/'
    :type cel_name2: str, optional
    :param root: Data directory
    :type root: str, optional
    :param embed_hydrogens: Set to True to keep hydrogens of molecules, 
        defaults to False
    :type embed_hydrogens: bool, optional
    :param rmsd_func: Backend to compute the RMSD, either ccdc or rdkit, 
        defaults to 'ccdc'
    :type rmsd_func: str, optional
    """
    
    def __init__(self,
                 rmsd_name: str = RMSD_DIRNAME,
                 cel_name1: str = GEN_CONF_DIRNAME,
                 cel_name2: str = BIO_CONF_DIRNAME,
                 root: str = DATA_DIRPATH,
                 embed_hydrogens: bool = False,
                 rmsd_func: str = 'ccdc') -> None:
        
        assert rmsd_func in ['rdkit', 'ccdc'], \
            'RMSD function must be rdkit or ccdc'
        
        self.rmsd_name = rmsd_name
        self.cel_name1 = cel_name1
        self.cel_name2 = cel_name2
        self.root = root
        self.embed_hydrogens = embed_hydrogens
        self.rmsd_func = rmsd_func
        
        self.rmsd_dir = os.path.join(self.root, rmsd_name)
        if not os.path.exists(self.rmsd_dir) :
            os.mkdir(self.rmsd_dir)
        self.cel_dir1 = os.path.join(self.root, self.cel_name1)
        self.cel_dir2 = os.path.join(self.root, self.cel_name2)
        
        self.cel_df_path = os.path.join(self.root, 
                                        self.cel_name2, 
                                        'ensemble_names.csv')
        self.cel_df = pd.read_csv(self.cel_df_path)
        
        self.mol_converter = MolConverter()
        
    
    def compute_rmsd_matrices(self) -> None:
        """Compute the RMSD matrices for all ligands in the libraries
        """
            
        params = list(zip(self.cel_df['ensemble_name'], self.cel_df['filename']))
            
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.compute_rmsd_matrix_thread, params)
        
        # for param in params:
        #     self.compute_rmsd_matrix_thread(param)
            
            
    def compute_rmsd_matrix_thread(self,
                                   params: Tuple[str, str]) -> None:
        """Compute the RMSD matric for one ligand

        :param params: Ensemble name and filename of the ligand
        :type params: Tuple[str, str]
        """
        name, filename = params
        file_prefix = filename.split('.')[0]
        new_filename = f'{file_prefix}.npy'
        filepath = os.path.join(self.rmsd_dir, new_filename)
        if not os.path.exists(filepath) :
            name, filename = params
            try:
                ce = ConfEnsembleLibrary.get_merged_ce(filename=filename, 
                                                       name=name,
                                                       root=self.root,
                                                       cel_name1=self.cel_name1,
                                                       cel_name2=self.cel_name2,
                                                       embed_hydrogens=self.embed_hydrogens)
                confs = [conf for conf in ce.mol.GetConformers()]
                gen_conf_ids = []
                bio_conf_ids = []
                for conf in confs :
                    conf_id = conf.GetId()
                    if conf.HasProp('PDB_ID') :
                        bio_conf_ids.append(conf_id)
                    else :
                        gen_conf_ids.append(conf_id)
                rmsd_matrix = self.get_rmsd_matrix(rdkit_mol=ce.mol, 
                                                    conf_ids1=gen_conf_ids, 
                                                    conf_ids2=bio_conf_ids)
                
                np.save(filepath, rmsd_matrix)
            except Exception as e:
                print(str(e))
                print(filename)
            
            
    def get_rmsd_matrix(self,
                        rdkit_mol: Mol,
                        conf_ids1: List[int],
                        conf_ids2: List[int]) -> np.ndarray:
        """Return the RMSD matrix between two set of conformation (based on 
        conf_ids) for the same molecule

        :param rdkit_mol: Input molecule
        :type rdkit_mol: Mol
        :param conf_ids1: Conf ids of generated conformers
        :type conf_ids1: List[int]
        :param conf_ids2: Conf ids of bioactive conformations
        :type conf_ids2: List[int]
        :raises RuntimeError: In case the rmsd_func is wrong
        :return: RMSD matrix of dimension (len(conf_ids1), len(conf_ids2))
        :rtype: np.ndarray
        """
        
        matrix_shape = (len(conf_ids1), len(conf_ids2))
        rmsd_matrix = np.zeros(matrix_shape)
        for i, conf_id1 in enumerate(conf_ids1) :
            for j, conf_id2 in enumerate(conf_ids2) :
                if self.rmsd_func == 'rdkit' :
                    rmsd = GetBestRMS(rdkit_mol, rdkit_mol, conf_id1, conf_id2)
                elif self.rmsd_func == 'ccdc' :
                    rmsd = self.get_ccdc_rmsd(rdkit_mol, conf_id1, conf_id2)
                else:
                    raise RuntimeError()
                rmsd_matrix[i, j] = rmsd
        return rmsd_matrix
         
            
    def get_ccdc_rmsd(self,
                      rdkit_mol: Mol,
                      conf_id1: int,
                      conf_id2: int) -> float:
        """Get the RMSD between two conformers of the same molecule using the 
        CSD Python API

        :param rdkit_mol: Input Molecule
        :type rdkit_mol: Mol
        :param conf_id1: First conf id
        :type conf_id1: int
        :param conf_id2: Second conf id
        :type conf_id2: int
        :return: RMSD
        :rtype: float
        """
        ccdc_mol1 = self.mol_converter.rdkit_conf_to_ccdc_mol(rdkit_mol, 
                                                                     conf_id1)
        ccdc_mol2 = self.mol_converter.rdkit_conf_to_ccdc_mol(rdkit_mol, 
                                                                     conf_id2)
        rmsd = MolecularDescriptors.rmsd(ccdc_mol1, ccdc_mol2, overlay=True)
        return rmsd
    
    
if __name__ == '__main__' :
    rc = RMSDCalculator()
    rc.compute_rmsd_matrices()
    