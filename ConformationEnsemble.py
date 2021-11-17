import unittest
import copy

from typing import List, Union
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Conformer
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms
from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
from numpy.testing import assert_array_equal
from tqdm import tqdm

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

class ConformationEnsemble(object) :
    
    """
    Class to store conformations from the same molecule in a single RDKit molecule
    Handles bioactive conformation as a property of RDKit Conformer
    Only handles molecule which have coordinates for hydrogens
    """
    
    def __init__(self, mol: Mol=None, mol_list: List[Mol]=None) :
        
        """
        Create a conformation ensemble; wrapping a RDKit mol
        Args :
            mol: Mol = RDKit molecule containing
            mol_list: List[Mol] = list of molecules of identical graph having each
                one conformer
        """
        
        if mol is not None :
            self.mol = mol
        elif mol_list is not None :
            if len(mol_list) == 1 :
                mol = mol_list[0]
                self.mol = mol
            else :
                self.mol = self.mol_from_list(mol_list)
        else :
            raise Exception('No input is given')
        
    def mol_from_list(self, mol_list: List[Mol]) :
        
        """
        Groups conformations coming from a list of RDKit molecule to a single 
        RDKit molecule with multiple Conformer
        Args :
            mol_list: List[Mol] = list of molecules having
        Returns :
            master_mol: Mol = single molecule grouping all conformations from input list
        """
        
        self._check_input_ensemble(mol_list)
        
        master_mol = mol_list[0]
        master_conf = master_mol.GetConformer()
        for new_mol in mol_list[1:] :
            dummy_mol = copy.deepcopy(new_mol)
            atom_map = master_mol.GetSubstructMatches(new_mol)
            
            for conf in dummy_mol.GetConformers() : # taking from dummy to avoid inplace change of Conformer object
                new_positions = conf.GetPositions()
                
                for master_atom_idx, new_atom_idx in enumerate(atom_map[0]) :
                    conf.SetAtomPosition(master_atom_idx, new_positions[new_atom_idx])
                
                new_conf_id = master_mol.AddConformer(conf, assignId=True)
            
        return master_mol
        
    def standardize_mol(self, mol: Mol) :
        
        """
        Depending on the input format, 2 mols might not have the same order of 
        atoms and bonds. This function is recreating the molecule using its smiles
        then associating the right coordinates for the conformations
        Args :
            mol: Mol = RDKit molecule to standardize
        Returns :
            new_mol: Mol = Standardized RDKit molecule with canonical atom/bond ordering
        """
        
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
        
    def add_conformations(self, mol: Mol) :
        
        """
        Add conformations to the existing RDKit Mol using a single RDKit Mol having one
        or multiple Conformer
        Args :
            mol: Mol = input molecule containing new conformations
        Returns :
            conf_ids: List = list of confId in the stored RDKit Mol corresponding
                to conformations added
        """
        
        self._check_conformation(mol)
        
        conf_ids = []
        for conf in mol.GetConformers() :
            conf_ids.append(self.mol.AddConformer(conf, assignId=True))
            
        return conf_ids
        
    def add_conformations_from_mol_list(self, mol_list: List[Mol]=None) :
        
        """
        Add conformations to the existing RDKit Mol from a list of RDKit Mol having
        one or multiple Conformer
        Args :
            mol_list: List[Mol] = list containing molecule having new conformations
        Returns :
            conf_ids: List = list of confId in the stored RDKit Mol corresponding
                to conformations added
        """
        
        self._check_input_ensemble(mol_list)
        
        conf_ids = []
        for mol in mol_list :
            conf_ids = conf_ids + self.add_conformations(mol)
            
        return conf_ids
        
    def add_bioactive_conformations(self, mol: Mol) :
        
        """
        Add bioactive conformations to the existing RDKit Mol using a single RDKit Mol 
        having one or multiple Conformer.
        Bioactive conformations will have a IsBioactive boolean property in
        corresponding Conformer in the RDKit Mol
        Args :
            mol: Mol = input molecule containing new bioactive conformations
        Returns :
            conf_ids: List = list of confId in the stored RDKit Mol corresponding
                to conformations added
        """
        
        conf_ids = self.add_conformations(mol)
        for conf_id in conf_ids :
            self.mol.GetConformer(conf_id).SetBoolProp(key='IsBioactive', val=True)
        return conf_ids
            
    def add_bioactive_conformations_from_mol_list(self, mol_list: List[Mol]=None) :
        
        """
        Add bioactive conformations to the existing RDKit Mol from a list of RDKit Mol 
        having one or multiple Conformer.
        Bioactive conformations will have a IsBioactive boolean property in
        corresponding Conformer in the RDKit Mol
        Args :
            mol_list: List[Mol] = list containing molecule having new conformations
        Returns :
            conf_ids: List = list of confId in the stored RDKit Mol corresponding
                to conformations added
        """
        
        self._check_input_ensemble(mol_list)
        conf_ids = []
        for mol in mol_list :
            conf_ids = conf_ids + self.add_bioactive_conformations(mol)
        return conf_ids
        
    def get_num_conformations(self) :
        return self.mol.GetNumConformers()
    
    def get_conformation_positions(self, i: int=0) :
        
        """
        Args :
            i: int = index of the Conformer in the RDKit Mol
        Returns :
            np.ndarray = coordinates of the atom in the RDKit Mol for the given Conformer index
        """
        
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
        input_smiles = Chem.MolToSmiles(mol)
        ensemble_smiles = Chem.MolToSmiles(self.mol)
        assert input_smiles == ensemble_smiles, f'Molecule {input_smiles} from input conformation is different to molecule {ensemble_smiles} from the ensemble'
            
    def _check_mol_has_conformation(self, mol) :
        assert mol.GetNumConformers()    
         
            
class ConformationEnsembleLibrary(object) :
    
    def __init__(self, mol_list: List[Mol]) :
        self.library = {}
        for mol in tqdm(mol_list) :
            smiles = Chem.MolToSmiles(mol)
            if smiles in self.library :
                self.library[smiles].add_conformations(mol)
            else :
                self.library[smiles] = ConformationEnsemble(mol=mol)
                
    def get_num_molecules(self) :
        return len(self.library)
    
    def get_conf_ensemble(self, smiles) :
        assert smiles in self.library, 'Input smiles not in library'
        return self.library[smiles]
    
    def get_unique_molecules(self) :
        return self.library.items()
            
    
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
        EmbedMultipleConfs(mol, 10)
        original_n_conf = self.conf_ensemble.get_num_conformations()
        conf_ids = self.conf_ensemble.add_conformations(mol)
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

    def test_conformation_ensemble_library(self):
        mol_list = []
        for smiles in ['CC(=O)NC1=CC=C(C=C1)O', 'CC(=O)NC1=CC=C(C=C1)O', 'C1=CC=CC=C1'] :
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol, addCoords=True)
            EmbedMultipleConfs(mol, 10)
            mol_list.append(mol)
            
        conf_ensemble_library = ConformationEnsembleLibrary(mol_list)
        self.assertEqual(conf_ensemble_library.get_num_molecules(), 2)
        self.assertEqual(conf_ensemble_library.get_conf_ensemble('[H]Oc1c([H])c([H])c(N([H])C(=O)C([H])([H])[H])c([H])c1[H]').get_num_conformations(), 20)
        self.assertEqual(conf_ensemble_library.get_conf_ensemble('[H]c1c([H])c([H])c([H])c([H])c1[H]').get_num_conformations(), 10)
        
if __name__ == '__main__':
    unittest.main()