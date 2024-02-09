import numpy as np

from .conf_ranker import ConfRanker
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from typing import List, Any


class EnergyRanker(ConfRanker):
    """Can rank conformers based on MMFF94s energy

    :param name: Name of the ranker
    :type name: str
    :param ascending: Set to True to rank conformer by ascending
        values, or False for descending values, defaults to True
    :type ascending: bool, optional
    """
    
    def __init__(self,
                 name: str = 'MMFF94s_energy',
                 ascending: bool = True) -> None:
        super().__init__(name=name, 
                         ascending=ascending)
        self.ascending = ascending


    def get_input_list_for_mol(self,
                               mol: Mol) -> List[Any]:
        """Return list of conformer energies

        :param mol: Input molecule
        :type mol: Mol
        :return: List of energies
        :rtype: List[Any]
        """
        
        # import pdb;pdb.set_trace()
        
        # Energy computed with hydrogens
        mol = Chem.AddHs(mol, addCoords=True)
        
        mol_properties = AllChem.MMFFGetMoleculeProperties(mol, 
                                                            mmffVariant='MMFF94s')
        
        input_list = []
        for conf in mol.GetConformers():
            conf_id = conf.GetId()
            force_field = AllChem.MMFFGetMoleculeForceField(mol, 
                                                            mol_properties,
                                                            confId=conf_id)
            energy = force_field.CalcEnergy()
            input_list.append(energy)
        
        return input_list


    def compute_values(self,
                        input_list: List[Any]
                        ) -> np.ndarray:
        """Compute values based on input_list. Here just a translation to np.ndarray

        :param input_list: Input list of energies
        :type input_list: List[Any]
        :return: Array of energies
        :rtype: np.ndarray
        """
        return np.array(input_list)
 