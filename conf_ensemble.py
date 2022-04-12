import unittest
import copy

from typing import List
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs

from collections.abc import Iterator

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

class ConfEnsemble(object) :
    
    """
    Class to store confs (conformations, also called conformers if low-energy) 
    from the same molecule in a single RDKit molecule
    """
    
    def __init__(self, 
                 mol: Mol=None, 
                 mol_list: List[Mol]=None, 
                 standardize_mols: bool=True,
                 embed_hydrogens: bool=False) :
        
        """
        Create a conf ensemble; wrapping a RDKit mol
        Args:
            mol: Mol = RDKit molecule containing
            mol_list: List[Mol] = list of molecules of identical graph having
                each one conformer
            embed_hydrogens: bool = Whether to include hydrogens in the 
                molecules (hydrogens are predicted for bioactive conformations 
                for instance)
        """
        
        self.standardize_mols = standardize_mols
        self.embed_hydrogens = embed_hydrogens
        if mol is not None :
            if self.standardize_mols :
                mol = self.standardize_mol(mol)
            self.mol = mol
        elif mol_list is not None :
            
            # if self.standardize_mols :
            #     standardized_mol_list = [self.standardize_mol(mol) 
            #                             for mol in mol_list]
            # else :
            #     standardized_mol_list = mol_list
                
            if len(mol_list) == 1 :
                mol = mol_list[0]
                if self.standardize_mol :
                    mol = self.standardize_mol(mol=mol)
                for conf in mol.GetConformers() : 
                    for prop in mol.GetPropNames() :
                        value = mol.GetProp(prop)
                        conf.SetProp(prop, str(value))
                self.mol = mol
            else :
                self.mol = self.mol_from_list(mol_list,
                                              standardize_mol=self.standardize_mols)
        else :
            raise Exception('No input given')
        
    def mol_from_list(self, 
                      mol_list: List[Mol],
                      standardize_mol: bool=True) :
        
        """
        Groups conformations coming from a list of RDKit molecule to a single 
        RDKit molecule with multiple Conformer
        Args:
            mol_list: List[Mol] = list of molecules having the same graph
        Returns:
            master_mol: Mol = single molecule grouping all conformations from 
                input list
        """
        if standardize_mol :
            standardized_mol_list = [self.standardize_mol(mol) for mol in mol_list]
            self._check_input_ensemble(standardized_mol_list)
        else :
            standardized_mol_list = mol_list
        
        #import pdb; pdb.set_trace()
        
        for i, mol in enumerate(standardized_mol_list) :
            if i == 0 :
                master_mol = copy.deepcopy(mol)
                conf = master_mol.GetConformer()
                for prop in mol.GetPropNames() :
                    value = mol.GetProp(prop)
                    conf.SetProp(prop, str(value))
            else :
                for conf in mol.GetConformers() : 
                    for prop in mol.GetPropNames() :
                        value = mol.GetProp(prop)
                        conf.SetProp(prop, str(value))
                    new_conf_id = master_mol.AddConformer(conf, assignId=True)
            
        return master_mol
        
    def standardize_mol(self, 
                        mol: Mol) :
        
        """
        Depending on the input format, 2 mols might not have the same order of 
        atoms and bonds. This function is recreating the molecule using its 
        smiles then reordering the conf coordinates to match the new atom 
        order
        Args:
            mol: Mol = RDKit molecule to standardize
        Returns:
            standard_mol: Mol = Standardized RDKit molecule with 
                canonical atom/bond ordering
        """
        if not self.embed_hydrogens :
            mol = Chem.RemoveHs(mol)
        
        self._check_mol_has_conf(mol)
        Chem.AssignAtomChiralTagsFromStructure(mol)
        standard_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        assert mol.GetNumAtoms() == standard_mol.GetNumAtoms()
        for prop in mol.GetPropNames() :
            value = mol.GetProp(prop)
            standard_mol.SetProp(prop, str(value))
            for conf in standard_mol.GetConformers() :
                conf.SetProp(prop, str(value))
        if self.embed_hydrogens :
            # MolFromSmiles is not embedding Hs by default
            standard_mol = Chem.AddHs(standard_mol, addCoords=True)
        else :
            self._check_mol_has_no_hydrogens(mol)
        atom_map = mol.GetSubstructMatches(standard_mol)
        # atom_map[standard_mol_atom_idx] = mol_atom_idx
        
        # loop to correct atom ordering of positions
        mol_copy = copy.deepcopy(mol)
        for conf in mol_copy.GetConformers() :
            #print(conf.GetPositions())
            #print(atom_map)
            new_positions = conf.GetPositions()[atom_map]
            #print(new_positions)
            for i, position in enumerate(new_positions) :
                conf.SetAtomPosition(i, position)
            #print(conf.GetPositions())
            
            for prop in mol.GetPropNames() :
                value = mol.GetProp(prop)
                conf.SetProp(prop, str(value))
            
            new_conf_id = standard_mol.AddConformer(conf, assignId=True)
        
        return standard_mol
        
    def add_confs_from_mol(self, 
                           mol: Mol,
                           standardize_mol: bool=True) :
        
        """
        Add conformations to the existing RDKit Mol using a single RDKit Mol 
        having one or multiple Conformer
        Args :
            mol: Mol = input molecule containing new conformations
        Returns :
            conf_ids: List = list of confId in the stored RDKit Mol 
                corresponding to conformations added
        """
        if standardize_mol :
            mol = self.standardize_mol(mol)
        
        self._check_conf(mol)
        
        conf_ids = []
        for conf in mol.GetConformers() :
            conf_ids.append(self.mol.AddConformer(conf, assignId=True))
            
        return conf_ids
        
    def add_confs_from_mol_list(self, mol_list: List[Mol]=None) :
        
        """
        Add conformations to the existing RDKit Mol from a list of RDKit Mol 
        having one or multiple Conformer
        Args :
            mol_list: List[Mol] = list containing molecule having new 
                conformations
        Returns :
            conf_ids: List = list of confId in the stored RDKit Mol 
                corresponding to conformations added
        """
        
        standardized_mol_list = [self.standardize_mol(mol) for mol in mol_list]
        self._check_input_ensemble(standardized_mol_list)
        
        conf_ids = []
        for mol in standardized_mol_list :
            conf_ids = conf_ids + self.add_confs_from_mol(mol)
            
        return conf_ids
        
    def get_num_confs(self) :
        return self.mol.GetNumConformers()
    
    def get_conf_positions(self, conf_id: int=0) :
        
        """
        Args :
            conf_id: int = index of the Conformer in the RDKit Mol
        Returns :
            np.ndarray = coordinates of the atom in the RDKit Mol for the given
                Conformer index
        """
        
        return self.mol.GetConformer(conf_id).GetPositions()
        
    def _check_input_ensemble(self, mol_list: List[Mol]) :
        
        """
        Checks if the molecules from the input ensemble are all the same
        (in case the user enters wrong input) and if there is at least 
        one conformer per mol
        Args :
            mol_list: List[Mol] = list of molecules of identical graph having 
                each one conformer
        """
        
        smiles_list = [Chem.MolToSmiles(mol) for mol in mol_list]
        assert len(set(smiles_list)) < 2, \
            'All molecules in the input list should be the same'
        
        assert all([mol.GetNumConformers() > 0 for mol in mol_list]), \
            'All molecules in the input should have at least one conformation'
        
    def _check_conf(self, mol: Mol) :
        
        """
        Checks if the input molecule is the same as the stored molecule
        Args :
            mol: Mol = input molecule
        """
        Chem.AssignAtomChiralTagsFromStructure(mol)
        Chem.AssignAtomChiralTagsFromStructure(self.mol)
        input_smiles = Chem.MolToSmiles(mol)
        ensemble_smiles = Chem.MolToSmiles(self.mol)
        assert input_smiles == ensemble_smiles, \
        f"""Molecule {input_smiles} from input conformation is different 
        to molecule {ensemble_smiles} from the ensemble"""
            
    def _check_mol_has_conf(self, mol) :
        assert mol.GetNumConformers(), 'Input molecule has no conf'
        
    def _check_mol_has_no_hydrogens(self, mol) :
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        assert 'H' not in atom_symbols, \
        """Input molecule has hydrogens, that are not embeded by the current 
        ConfEnsemble"""
         
    
class ConfEnsembleTest(unittest.TestCase):

    def setUp(self):
        self.smiles = 'CC(=O)NC1=CC=C(C=C1)O'
        mols = [Chem.MolFromSmiles(self.smiles) for _ in range(2)]
        mols = [Chem.AddHs(mol, addCoords=True) for mol in mols]
        for mol in mols :
            EmbedMolecule(mol)
        mols = [Chem.RemoveHs(mol) for mol in mols]
        self.conf_ensemble = ConfEnsemble(mol_list=mols)
    
    def test_add_confs_from_mol(self):
        mol = Chem.MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O')
        mol = Chem.AddHs(mol, addCoords=True)
        EmbedMultipleConfs(mol, 10)
        mol = Chem.RemoveHs(mol)
        original_n_conf = self.conf_ensemble.get_num_confs()
        conf_ids = self.conf_ensemble.add_confs_from_mol(mol)
        new_n_conf = self.conf_ensemble.get_num_confs()
        self.assertEqual(new_n_conf, original_n_conf + mol.GetNumConformers())
        
#         for i, conf in enumerate(mol.GetConformers()) :
#             print(self.conf_ensemble.get_conf_positions(conf_ids[i]))
#             print(conf.GetPositions())
#             assert_array_equal(self.conf_ensemble.get_conf_positions(conf_ids[i]), 
#                                conf.GetPositions())
        
    def test_add_confs_from_mol_list(self):
        mols = [Chem.MolFromSmiles(self.smiles) for _ in range(2)]
        mols = [Chem.AddHs(mol, addCoords=True) for mol in mols]
        for mol in mols :
            EmbedMolecule(mol)
        mols = [Chem.RemoveHs(mol) for mol in mols]
        original_n_conf = self.conf_ensemble.get_num_confs()
        conf_ids = self.conf_ensemble.add_confs_from_mol_list(mols)
        new_n_conf = self.conf_ensemble.get_num_confs()
        n_conf_added = sum([mol.GetNumConformers() for mol in mols])
        self.assertEqual(new_n_conf, original_n_conf + n_conf_added)
        
#         assert_array_equal(self.conf_ensemble.get_conf_positions(new_n_conf - 1), 
#                            mol.GetConformer().GetPositions())
        
    def test_add_confs_wrong_molecule(self):
        mol = Chem.MolFromSmiles('C1=CC=CC=C1')
        mol = Chem.AddHs(mol, addCoords=True)
        EmbedMolecule(mol)
        mol = Chem.RemoveHs(mol)
        with self.assertRaises(AssertionError):
            self.conf_ensemble.add_confs_from_mol(mol)
        
if __name__ == '__main__':
    unittest.main()