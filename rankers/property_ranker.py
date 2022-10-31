import numpy as np

from .conf_ranker import ConfRanker
from rdkit.Chem.rdchem import Mol
from typing import List, Any, Sequence

class PropertyRanker(ConfRanker):
    
    def __init__(self,
                 descriptor_name: str = 'E',
                 ascending: bool = True) -> None:
        super().__init__(name=descriptor_name, 
                         ascending=ascending)
        self.descriptor_name = descriptor_name
        self.ascending = ascending


    def get_input_list_from_mol(self,
                               mol: Mol) -> List[Any]:
        input_list = [conf for conf in mol.GetConformers()]
        return input_list


    def get_values(self,
                   input_list) -> Sequence[float]:
        try:
            confs = input_list
            values = []
            for conf in confs:
                # prop_names = [prop_name for prop_name in conf.GetPropNames()] # we have to iterate because GetPropNames returns a "vector" object
                # assert self.descriptor_name in prop_names
                value = conf.GetProp(self.descriptor_name)
                if isinstance(value, str):
                    value = value.strip()
                    value = float(value)
                values.append(value)
        except:
            import pdb;pdb.set_trace()
        return np.array(values)
 
 
    def get_input_list_from_data_list(self,
                                      data_list: Sequence[Any]) -> Sequence[Any]:
        input_list = [conf for conf, rmsd in data_list]
        return input_list
    
    def get_output_list_from_data_list(self,
                                      data_list: Sequence[Any]) -> Sequence[Any]:
        output_list = [rmsd for conf, rmsd in data_list]
        return output_list