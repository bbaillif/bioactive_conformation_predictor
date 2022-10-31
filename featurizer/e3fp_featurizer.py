import numpy as np
import torch

from .mol_featurizer import MolFeaturizer
from rdkit.Chem.rdchem import Mol
from e3fp.fingerprint.generate import fprints_dict_from_mol
from e3fp.fingerprint.db import FingerprintDatabase, concat
from e3fp.fingerprint.fprint import CountFingerprint
from scipy.sparse import csr_matrix

class E3FPFeaturizer(MolFeaturizer):
    
    def __init__(self,
                 n_bits: int = 4096,
                 level: int = 5,
                 counts: bool = True) -> None:
        self.n_bits = n_bits
        self.level = level
        self.counts = counts
    
    
    def featurize_mol(self,
                      mol: Mol) -> torch.Tensor:
        fp_db = self.get_fp_db(mol)
        array = self.get_array_from_db(fp_db)
        return torch.tensor(array, dtype=torch.float32)
    
    
    def get_array_from_db(self,
                          db: FingerprintDatabase):
        array = csr_matrix.toarray(db.array)
        array = np.int16(array).squeeze()
        return array
        
        
    def get_fp_db(self,
                  mol: Mol):
        
        fp_dict = fprints_dict_from_mol(mol, 
                                        bits=self.n_bits, 
                                        level=self.level, 
                                        first=-1, 
                                        counts=True)
        fps = fp_dict[self.level]
        
        fp_db = FingerprintDatabase(fp_type=CountFingerprint, 
                                level=self.level)
        fp_db.add_fingerprints(fps)
        return fp_db
        
        