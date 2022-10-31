import numpy as np

from .conf_ranker import ConfRanker
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from typing import List, Any, Sequence


class EnergyRanker(ConfRanker):
    
    def __init__(self,
                 name: str = 'MMFF94s_energy',
                 ascending: bool = True) -> None:
        super().__init__(name=name, 
                         ascending=ascending)
        self.ascending = ascending


    def get_input_list_from_mol(self,
                               mol: Mol) -> List[Any]:
        input_list = []
        for conf in mol.GetConformers():
            conf_id = conf.GetId()
            mol_properties = AllChem.MMFFGetMoleculeProperties(mol, 
                                                               mmffVariant='MMFF94s')
            force_field = AllChem.MMFFGetMoleculeForceField(mol, 
                                                        mol_properties,
                                                        confId=conf_id)
            input_list.append(force_field)
        return input_list


    def get_values(self,
                   input_list) -> Sequence[float]:
        force_fields = input_list
        values = [force_field.CalcEnergy() for force_field in force_fields]
        return np.array(values)
 