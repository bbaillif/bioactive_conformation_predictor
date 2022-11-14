import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit.Chem.rdMolAlign import GetBestRMS
from typing import List
from rdkit.Chem.rdchem import Mol
from multiprocessing import Pool
from conf_ensemble import ConfEnsemble
from ccdc_rdkit_connector import CcdcRdkitConnector
from ccdc.descriptors import MolecularDescriptors

class RMSDCalculator() :
    
    def __init__(self,
                 rmsd_name: str = 'rmsds',
                 cel_name1: str = 'gen_conf_ensembles/',
                 cel_name2: str = 'pdb_conf_ensembles/',
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/') -> None:
        self.rmsd_name = rmsd_name
        self.cel_name1 = cel_name1
        self.cel_name2 = cel_name2
        self.root = root
        
        self.rmsd_dir = os.path.join(self.root, rmsd_name)
        if not os.path.exists(self.rmsd_dir) :
            os.mkdir(self.rmsd_dir)
        self.cel_dir1 = os.path.join(self.root, self.cel_name1)
        self.cel_dir2 = os.path.join(self.root, self.cel_name2)
        
        self.cel_df_path = os.path.join(self.root, 
                                        self.cel_name2, 
                                        'ensemble_names.csv')
        self.cel_df = pd.read_csv(self.cel_df_path)
        
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
        
    
    def compute_rmsd_matrices(self) :
            
        params = list(zip(self.cel_df['ensemble_name'], self.cel_df['filename']))
            
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.compute_rmsd_matrix_thread, params)
        
        # for param in params:
        #     self.compute_rmsd_matrix_thread(param)
            
            
    def compute_rmsd_matrix_thread(self,
                                   params) :
        name, filename = params
        file_prefix = filename.split('.')[0]
        new_filename = f'{file_prefix}.npy'
        filepath = os.path.join(self.rmsd_dir, new_filename)
        if not os.path.exists(filepath) :
            name, filename = params
            try:
                ce = self.get_merged_ce(filename, name)
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
                        conf_ids2: List[int],
                        rmsd_func: str = 'rdkit') -> np.ndarray:
        matrix_shape = (len(conf_ids1), len(conf_ids2))
        rmsd_matrix = np.zeros(matrix_shape)
        for i, conf_id1 in enumerate(conf_ids1) :
            for j, conf_id2 in enumerate(conf_ids2) :
                if rmsd_func == 'rdkit' :
                    rmsd = GetBestRMS(rdkit_mol, rdkit_mol, conf_id1, conf_id2)
                else :
                    rmsd = self.get_ccdc_rmsd(rdkit_mol, conf_id1, conf_id2)
                rmsd_matrix[i, j] = rmsd
        return rmsd_matrix
         
            
    def get_ccdc_rmsd(self,
                      rdkit_mol: Mol,
                      conf_id1: int,
                      conf_id2: int) -> float:
        ccdc_mol1 = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol, 
                                                                     conf_id1)
        ccdc_mol2 = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol, 
                                                                     conf_id2)
        rmsd = MolecularDescriptors.rmsd(ccdc_mol1, ccdc_mol2, overlay=True)
        return rmsd
    
    
    def get_merged_ce(self,
                      filename: str,
                      name: str) -> ConfEnsemble:
        
        ce_filepath1 = os.path.join(self.cel_dir1, filename)
        ce1 = ConfEnsemble.from_file(filepath=ce_filepath1, 
                                                name=name)
        
        ce_filepath2 = os.path.join(self.cel_dir2, filename)
        ce2 = ConfEnsemble.from_file(filepath=ce_filepath2, 
                                                    name=name)
        
        ce1.add_mol(ce2.mol, standardize=False)
        return ce1
    
if __name__ == '__main__' :
    rc = RMSDCalculator()
    rc.compute_rmsd_matrices()
    