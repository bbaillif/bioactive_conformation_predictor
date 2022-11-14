import os
import numpy as np
import pandas as pd
import torch
import shutil

from tqdm import tqdm
from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset
from e3fp.fingerprint.generate import fprints_dict_from_mol
from e3fp.fingerprint.db import FingerprintDatabase, concat
from e3fp.fingerprint.fprint import CountFingerprint
from scipy.sparse import csr_matrix
from .split.data_split import DataSplit
from multiprocessing import Pool
from .conf_ensemble_dataset import ConfEnsembleDataset

#TODO: Documentation

class E3FPDataset(ConfEnsembleDataset, Dataset):
    
    def __init__(self,
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/',
                 n_bits: int = 4096,
                 level: int = 5) -> None:
        
        ConfEnsembleDataset.__init__(self, root=root)
        
        self.n_bits = n_bits
        self.level = level
        
        self.mol_id_df_filename = f'e3fp_mol_ids.csv'
        self.mol_id_df_path = os.path.join(self.root, self.mol_id_df_filename)
        
        self.e3fp_dir = os.path.join(self.root, 'e3fp/')
        if not os.path.exists(self.e3fp_dir):
            os.mkdir(self.e3fp_dir)
        if not os.path.exists(self.mol_id_df_path) :
            self.process()
        self.load()
                
                
    def process(self):
        params = []
        for i, row in tqdm(self.cel_df.iterrows(), total=self.cel_df.shape[0]) :
            name = row['ensemble_name']
            filename = row['filename']
            params.append((name, filename))
            
        # for param in params:
        #     self.compute_fp_thread(param[0], param[1])
            
        with Pool(processes=20) as pool:
            results = pool.starmap(self.compute_fp_thread, params)
           
        all_mol_ids = [] 
        for mol_ids in results:
            all_mol_ids.extend(mol_ids)
            
        self.mol_id_df = pd.DataFrame({'mol_id' : all_mol_ids})
        self.mol_id_df.to_csv(self.mol_id_df_path)
                
                
    def load(self):
        dbs = []
        for filename in tqdm(self.cel_df['filename'].values) :
            fpz_filename = filename.replace('.sdf', '.fpz')
            fpz_filepath = os.path.join(self.e3fp_dir, fpz_filename)
            if os.path.exists(fpz_filepath):
                db = FingerprintDatabase.load(fpz_filepath)
                dbs.append(db)
        self.db = concat(dbs)
        self.mol_id_df = pd.read_csv(self.mol_id_df_path, index_col=0)
                
                
    def compute_fp_thread(self, 
                          name: str, 
                          filename: str) -> None:
        # name, filename = params
        fpz_filename = filename.replace('.sdf', '.fpz')
        fpz_filepath = os.path.join(self.e3fp_dir, fpz_filename)
        try:
            ce = self.get_merged_ce(filename, name) # Merge bio + conf ensembles
            mol = ce.mol
                
            mol_ids = self.compute_mol_ids(mol, name) # Give ids to recognize each conf
            
            if not os.path.exists(fpz_filepath):
                fp_dict = fprints_dict_from_mol(mol, 
                                            bits=self.n_bits, 
                                            level=self.level, 
                                            first=-1, 
                                            counts=True)
                fps = fp_dict[self.level]
                
                db = FingerprintDatabase(fp_type=CountFingerprint, 
                                        name=name, 
                                        level=self.level)
                db.add_fingerprints(fps)
                db.savez(fpz_filepath)
            
        except:
            print(f'{name} failed')
            mol_ids = []
            
        return mol_ids
        
        
    def add_bioactive_rmsds(self,
                            data_split: DataSplit) -> None:
        rmsds = self.get_bioactive_rmsds(data_split)
        self.rmsds = torch.tensor(rmsds, dtype=torch.float32)
        
    
    def __getitem__(self, 
                    index) -> Tuple[Tensor, Tensor]:
        csr_array = self.db.array[index]
        array = csr_matrix.toarray(csr_array)
        array = np.int16(array).squeeze()
        fp = torch.tensor(array, dtype=torch.float32)
        if hasattr(self, 'rmsds'):
            rmsd = self.rmsds[index]
        else:
            rmsd = 0
        return fp, rmsd
        
    
    def __len__(self) -> int:
        return self.db.array.shape[0]