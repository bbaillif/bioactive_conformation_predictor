import unittest

from typing import List, Union
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Conformer
from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
from numpy.testing import assert_array_equal

class ConformationEnsemble(object) :
    
    """
    Class to store a list of RDKit molecules as a conformation ensemble
    Handles bioactive conformation as a property
    """
    
    def __init__(self, mol: Mol=None, mol_list: List[Mol]=None) :
        
        """
        Create a conformation ensemble; wrapping a RDKit mol
        Args :
            mol_list: List[Mol] = list of molecules of identical graph having each
                one conformer
        """
        
        if mol is not None :
            self.mol = self.standardize_mol(mol)
        elif mol_list is not None :
            if len(mol_list) == 1 :
                mol = mol_list[0]
                self.mol = self.standardize_mol(mol)
            else :
                self.mol = self.mol_from_list(mol_list)
        else :
            raise Exception('No input is given')
        
    def mol_from_list(self, mol_list) :
        
        self._check_input_ensemble(mol_list)
        
        master_mol = mol_list[0]
        for mol in mol_list[:1] :
            new_mol = self.standardize_mol(mol)
            for conf in new_mol.GetConformers() :
                master_mol.AddConformer(conf, assignId=True)
            
        return master_mol
        
    def standardize_mol(self, mol) :
        
        self._check_mol_has_conformation(mol)
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) # Ensure we have a canonical atom/bond ordering
        new_mol = Chem.AddHs(new_mol, addCoords=True) # Because MolFromSmiles is not embedding Hs by default
        atom_map = mol.GetSubstructMatches(new_mol) # atom_map[new_mol_atom_idx] = mol_atom_idx
        
        # loop to correct atom ordering of positions
        for conf in mol.GetConformers() :
            new_mol.AddConformer(conf, assignId=True)
            new_positions = conf.GetPositions()[atom_map]
            last_conf = new_mol.GetConformers()[-1]
            for i in range(len(new_positions)) :
                last_conf.SetAtomPosition(i, new_positions[i])
        
        return new_mol
        
    def add_conformations(self, mol: Mol, return_conf_ids=False) :
        
        self._check_conformation(mol)
        
        conf_ids = []
        for conf in mol.GetConformers() :
            conf_ids.append(self.mol.AddConformer(conf, assignId=True))
            
        if return_conf_ids :
            return conf_ids
        
    def add_conformations_from_mol_list(self, mol_list: List[Mol]=None, return_conf_ids=False) :
        
        self._check_input_ensemble(mol_list)
        conf_ids = []
        for mol in mol_list :
            conf_ids = conf_ids + self.add_conformations(mol, return_conf_ids=True)
        
    def add_bioactive_conformations(self, mol: Mol) :
        
        conf_ids = self.add_conformations(mol, return_conf_ids=True)
        for conf_id in conf_ids :
            self.mol.GetConformer(conf_id).SetBoolProp(key='IsBioactive', val=True)
            
    def add_bioactive_conformations_from_mol_list(self, mol_list: List[Mol]=None) :
        
        self._check_input_ensemble(mol_list)
        for mol in mol_list :
            self.add_bioactive_conformations(mol)
        
    def get_num_conformations(self) :
        return self.mol.GetNumConformers()
    
    def get_conformation_positions(self, i: int=0) :
        return self.mol.GetConformer(i).GetPositions()
        
    def _check_input_ensemble(self, mol_list: List[Mol]) :
        
        """
        Checks if the molecules from the input ensemble are all the same
        (in case the user enters wrong input) and if there is at least one conformer per mol
        Args :
            mol_list: List[Mol] = list of molecules of identical graph having each
                one conformer
        """
        
        smiles_list = [Chem.MolToSmiles(mol) for mol in mol_list]
        assert len(set(smiles_list)) < 2, 'All molecules in the input list should be the same'
        
        assert all([mol.GetNumConformers() > 0 for mol in mol_list]), 'All molecules in the input should have at least one conformation'
        
    def _check_conformation(self, mol: Mol) :
        
        """
        Checks if the molecules from the input ensemble are all the same
        (in case the user enters wrong input)
        Args :
            mol_list: List[Mol] = list of molecules of identical graph having each
                one conformer
        """
        
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(self.mol), 'Molecule from input conformation is different to molecule from the ensemble'
            
    def _check_mol_has_conformation(self, mol) :
        assert mol.GetNumConformers()    
            
    
class ConformationEnsembleTest(unittest.TestCase):

    def setUp(self):
        self.smiles = 'CC(=O)NC1=CC=C(C=C1)O'
        mols = [Chem.MolFromSmiles(self.smiles) for _ in range(2)]
        mols = [Chem.AddHs(mol, addCoords=True) for mol in mols]
        for mol in mols :
            EmbedMolecule(mol)
        self.conf_ensemble = ConformationEnsemble(mol_list=mols)
    
    def test_add_conformations(self):
        mol = Chem.MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O')
        mol = Chem.AddHs(mol, addCoords=True)
        EmbedMultipleConfs(mol)
        original_n_conf = self.conf_ensemble.get_num_conformations()
        conf_ids = self.conf_ensemble.add_conformations(mol, return_conf_ids=True)
        new_n_conf = self.conf_ensemble.get_num_conformations()
        self.assertEqual(new_n_conf, original_n_conf + mol.GetNumConformers())
        
        for i, conf in enumerate(mol.GetConformers()) :
            assert_array_equal(self.conf_ensemble.get_conformation_positions(conf_ids[i]), conf.GetPositions())
        
    def test_add_conformations_from_mol_list(self):
        mols = [Chem.MolFromSmiles(self.smiles) for _ in range(2)]
        mols = [Chem.AddHs(mol, addCoords=True) for mol in mols]
        for mol in mols :
            EmbedMolecule(mol)
        original_n_conf = self.conf_ensemble.get_num_conformations()
        conf_ids = self.conf_ensemble.add_conformations_from_mol_list(mols)
        new_n_conf = self.conf_ensemble.get_num_conformations()
        n_conf_added = sum([mol.GetNumConformers() for mol in mols])
        self.assertEqual(new_n_conf, original_n_conf + n_conf_added)
        
        assert_array_equal(self.conf_ensemble.get_conformation_positions(new_n_conf - 1), mol.GetConformer().GetPositions())
        
    def test_add_conformations_wrong_molecule(self):
        mol = Chem.MolFromSmiles('C1=CC=CC=C1')
        mol = Chem.AddHs(mol, addCoords=True)
        EmbedMolecule(mol)
        with self.assertRaises(AssertionError):
            self.conf_ensemble.add_conformations(mol)
            

    def test_add_bioactive_conformation(self):
        mol = Chem.MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O')
        mol = Chem.AddHs(mol, addCoords=True)
        EmbedMolecule(mol)
        original_n_conf = self.conf_ensemble.get_num_conformations()
        self.conf_ensemble.add_bioactive_conformations(mol)
        self.assertEqual(self.conf_ensemble.get_num_conformations(), original_n_conf + 1)
        #self.assertEqual(self.conf_ensemble.bioactive_conformation_indices, [original_n_conf])

if __name__ == '__main__':
    unittest.main()