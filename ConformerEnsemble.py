import unittest

from typing import List
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import EmbedMolecule
from numpy.testing import assert_array_equal

class ConformationEnsemble(object) :
    
    def __init__(self, mol_list: List[Mol]=[]) :
        
        self.ensemble = mol_list
        self.bioactive_conformation_indices = []
        
    def add_conformation(self, mol: Mol) :
        
        self._check_conformation(mol)
        self.ensemble.append(mol)
        
    def add_bioactive_conformation(self, mol: Mol) :
        
        self._check_conformation(mol)
        self.bioactive_conformation_indices.append(len(self.ensemble))
        self.add_conformation(mol)
        
    def get_num_conformations(self) :
        return len(self.ensemble)
    
    def get_conformation_positions(self, i: int=-1) :
        return self.ensemble[i].GetConformer().GetPositions()
        
    def _check_conformation(self, mol: Mol) :
        
        if self.ensemble :
            assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(self.ensemble[0]), 'Molecule from input conformation is different to molecule from the ensemble'
            
    
class ConformationEnsembleTest(unittest.TestCase):

    def setUp(self):
        self.conf_ensemble = ConformationEnsemble()
    
    def test_add_conformation(self):
        mol = Chem.MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O')
        EmbedMolecule(mol)
        original_n_conf = self.conf_ensemble.get_num_conformations()
        self.conf_ensemble.add_conformation(mol)
        self.assertEqual(self.conf_ensemble.get_num_conformations(), original_n_conf + 1)
        assert_array_equal(self.conf_ensemble.get_conformation_positions(-1), mol.GetConformer().GetPositions())

    def test_add_bioactive_conformation(self):
        mol = Chem.MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O')
        EmbedMolecule(mol)
        original_n_conf = self.conf_ensemble.get_num_conformations()
        self.conf_ensemble.add_bioactive_conformation(mol)
        self.assertEqual(self.conf_ensemble.get_num_conformations(), original_n_conf + 1)
        self.assertEqual(self.conf_ensemble.bioactive_conformation_indices, [0])

if __name__ == '__main__':
    unittest.main()