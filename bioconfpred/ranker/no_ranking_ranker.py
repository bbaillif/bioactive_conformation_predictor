import numpy as np

from .conf_ranker import ConfRanker
from rdkit.Chem.rdchem import Mol
from typing import List, Any, Sequence

class NoRankingRanker(ConfRanker):
    """Fake ranker that don't apply any ranking

    :param name: Name of the ranker
    :type name: str
    :param ascending: Set to True to rank conformer by ascending
        values, or False for descending values, defaults to True
    :type ascending: bool, optional
    """
    
    def __init__(self, 
                 name: str = 'NoRanking', 
                 ascending: bool = True):
        super().__init__(name, ascending)
        
    def get_input_list_for_mol(self,
                                mol: Mol) -> List[Any]:
        return np.array(range(mol.GetNumConformers()))

    def compute_values(self,
                   input_list) -> Sequence[float]:
        return input_list