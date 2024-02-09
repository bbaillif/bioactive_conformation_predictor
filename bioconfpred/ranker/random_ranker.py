import numpy as np

from .conf_ranker import ConfRanker
from rdkit.Chem import Mol
from typing import List, Any, Sequence

class RandomRanker(ConfRanker):
    """Randomly rank conformers

    :param name: Name of the ranker
    :type name: str
    :param ascending: Set to True to rank conformer by ascending
        values, or False for descending values, defaults to True
    :type ascending: bool, optional
    """
    
    def __init__(self, 
                 name: str = 'Shuffle', 
                 ascending: bool = True):
        super().__init__(name, ascending)
        
    def get_input_list_for_mol(self,
                                mol: Mol) -> List[Any]:
        n_confs = mol.GetNumConformers()
        return np.random.randn(n_confs)

    def compute_values(self,
                       input_list) -> Sequence[float]:
        return input_list