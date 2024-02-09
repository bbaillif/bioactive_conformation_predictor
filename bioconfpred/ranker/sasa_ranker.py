import numpy as np

from .conf_ranker import ConfRanker
from rdkit import Chem
from rdkit.Chem.rdFreeSASA import CalcSASA, classifyAtoms
from rdkit.Chem import Mol
from typing import List, Any, Sequence


class SASARanker(ConfRanker):
    """Rank conformers using the Solvent Accessible Surface Area (SASA)

    :param name: Name of the ranker
    :type name: str
    :param ascending: Set to True to rank conformer by ascending
        values, or False for descending values, defaults to False
    :type ascending: bool, optional
    """
    
    def __init__(self,
                 name: str = 'SASA',
                 ascending: bool = False) -> None:
        super().__init__(name=name, 
                         ascending=ascending)
        self.ascending = ascending


    def get_input_list_for_mol(self,
                               mol: Mol) -> List[Any]:
        
        # import pdb;pdb.set_trace()
        
        # SASA computed with hydrogens
        mol = Chem.AddHs(mol, addCoords=True)
        
        radiis = classifyAtoms(mol)
        
        input_list = []
        for conf in mol.GetConformers():
            conf_id = conf.GetId()
            sasa = CalcSASA(mol, radiis, confIdx=conf_id)
            input_list.append(sasa)
        
        return input_list


    def compute_values(self,
                   input_list) -> Sequence[float]:
        return np.array(input_list)
 