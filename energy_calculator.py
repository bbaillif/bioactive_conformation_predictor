import numpy as np

from abc import abstractmethod
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import SDWriter
from xtb.ase.calculator import XTB

class EnergyCalculator() :
    
    @abstractmethod
    def get_energy(self, 
                   mol: Mol, 
                   conf_id: int=0) -> float :
        raise NotImplementedError
        
        
class XtbEnergyCalculator(EnergyCalculator) :
        
    def get_energy(self,
                   mol: Mol,
                   conf_id: int=0) -> float :
        sdf_filename = 'mol.sdf'
        with SDWriter(sdf_filename) as sdwriter :
            sdwriter.write(Chem.AddHs(mol, addCoords=True))
        ase_mol = read(sdf_filename)
        ase_mol.calc = XTB(method="GFN2-xTB")
        energy = ase_mol.get_potential_energy()
        return energy
    
    
class UFFCalculator(EnergyCalculator) :
    
    def get_energy(self, 
                   mol: Mol,
                   conf_id: int=0) -> float :
        mol = Chem.AddHs(mol, addCoords=True)
        uff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        energy = uff.CalcEnergy()
        return energy