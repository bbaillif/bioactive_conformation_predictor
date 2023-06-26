import numpy as np

from .conf_ranker import ConfRanker
from rdkit import Chem
from rdkit.Chem.Descriptors3D import RadiusOfGyration
from rdkit.Chem.rdchem import Mol
from typing import List, Any, Sequence


class RGyrRanker(ConfRanker):
    """Rank conformers using the Radius of Gyration (RGyr)

    :param name: Name of the ranker
    :type name: str
    :param ascending: Set to True to rank conformer by ascending
        values, or False for descending values, defaults to False
    :type ascending: bool, optional
    """
    
    def __init__(self,
                 name: str = 'RGyr',
                 ascending: bool = False) -> None:
        super().__init__(name=name, 
                         ascending=ascending)
        self.ascending = ascending


    def get_input_list_for_mol(self,
                               mol: Mol) -> List[Any]:
        
        # import pdb;pdb.set_trace()
        
        # RGyr computed with hydrogens
        mol = Chem.AddHs(mol, addCoords=True)
        
        input_list = []
        for conf in mol.GetConformers():
            conf_id = conf.GetId()
            sasa = RadiusOfGyration(mol, confId=conf_id)
            input_list.append(sasa)
        
        return input_list


    def compute_values(self,
                   input_list) -> Sequence[float]:
        return np.array(input_list)
 