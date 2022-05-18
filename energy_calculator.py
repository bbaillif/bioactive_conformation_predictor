import numpy as np

from abc import abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from xtb.libxtb import VERBOSITY_MINIMAL
from xtb.interface import Calculator, Param

class EnergyCalculator() :
    
    @abstractmethod
    def get_energy(self, 
                   mol: Mol, 
                   conf_id: int=0) -> float :
        raise NotImplementedError
        
        
class XtbEnergyCalculator(EnergyCalculator) :
    
    def __init__(self) -> None:
        self.verbose = VERBOSITY_MINIMAL
        
    def get_energy(self,
                   mol: Mol,
                   conf_id: int=0) -> float :
        # mol = Chem.AddHs(mol, addCoords=True)
        numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        numbers = np.array(numbers)
        conf = mol.GetConformer(conf_id)
        positions = conf.GetPositions()
        calc = Calculator(Param.GFN2xTB, numbers, positions)
        calc.set_verbosity(self.verbose)
        res = calc.singlepoint()
        energy = res.get_energy()
        return energy
    
    
class UFFCalculator(EnergyCalculator) :
    
    def get_energy(self, 
                   mol: Mol,
                   conf_id: int=0) -> float :
        mol = Chem.AddHs(mol, addCoords=True)
        uff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        energy = uff.CalcEnergy()
        return energy