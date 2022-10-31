import numpy as np

from .conf_ranker import ConfRanker
from rdkit.Chem.rdchem import Mol
from typing import List, Any, Sequence

class ShuffleRanker(ConfRanker):
    
    def __init__(self, 
                 name: str = 'shuffle', 
                 ascending: bool = True):
        super().__init__(name, ascending)
        
    def get_input_list_from_mol(self,
                                mol: Mol) -> List[Any]:
        n_confs = mol.GetNumConformers()
        return np.random.randn(n_confs)

    def get_values(self,
                   input_list) -> Sequence[float]:
        return input_list