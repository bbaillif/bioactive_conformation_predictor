import unittest
import pickle
import os

from rdkit import Chem
from rdkit.Chem.rdchem import (
    Atom, 
    Bond, 
    RingInfo, 
    Mol)

from conf_ensemble import ConfEnsembleLibrary
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

    
class MoleculeEncoders(object) :
    
    """
    Build one hot encoders for molecule feature extraction functions, based on 
    available values in a conformation ensemble library. A dict is used to 
    store OneHotEncoder objects using the function name (because RDKit functions
    are from Boost, and therefore not picklable)
    """
    
    def __init__(self) :
        self.encoders = {}
        self.encoded_atom_function_names = [f'{function.__module__}.{function.__name__}' 
                                            for function in [Atom.GetAtomicNum, 
                                                            Atom.GetDegree, 
                                                            Atom.GetHybridization, 
                                                            Atom.GetChiralTag, 
                                                            Atom.GetImplicitValence, 
                                                            Atom.GetFormalCharge]]
        self.encoded_bond_function_names = [f'{function.__module__}.{function.__name__}' 
                                            for function in [Bond.GetBondType]]
        self.encoded_ring_function_names = [f'{function.__module__}.{function.__name__}' 
                                            for function in [RingInfo.NumAtomRings]]
    
    def create_encoders(self, 
                        conf_ensemble_library: ConfEnsembleLibrary) :
        all_function_names = self.encoded_atom_function_names + self.encoded_bond_function_names + self.encoded_ring_function_names
        for function_name in tqdm(all_function_names) :
            self.create_one_hot_encoder(function_name=function_name, 
                                        conf_ensemble_library=conf_ensemble_library)
    
    def create_one_hot_encoder(self, 
                               function_name: str, 
                               conf_ensemble_library: ConfEnsembleLibrary) :
        function = eval(function_name)
        values = self.get_all_function_values_in_library(function_name=function_name,
                                                                conf_ensemble_library=conf_ensemble_library)
        self.encoders[function_name] = OneHotEncoder(sparse=False).fit([[value] for value in values])
    
    def encode(self, value, function_name: str) :
        return self.encoders[function_name].transform([value])
    
    def __getitem__(self, function_name: str) :
        assert function_name in self.encoders, 'The function you are asking is not encoded, please use create_one_hot_encoder'
        return self.encoders[function_name]
    
    def get_encoder_categories(self, function_name: str) :
        return self.encoders[function_name].categories_
    
    def get_all_function_values_in_library(self, function_name: str, conf_ensemble_library: ConfEnsembleLibrary) :
        function = eval(function_name)
        values = []
        for smiles, conf_ensemble in conf_ensemble_library.get_unique_molecules() :
            if function.__module__ == 'Atom' : 
                values.extend([function(atom) for atom in conf_ensemble.mol.GetAtoms()])
            elif function.__module__ == 'Bond' :
                values.extend([function(bond) for bond in conf_ensemble.mol.GetBonds()])
            elif function.__module__ == 'RingInfo' :
                ring_info = conf_ensemble.mol.GetRingInfo()
                values.extend([function(ring_info, atom_idx) for atom_idx, atom in enumerate(conf_ensemble.mol.GetAtoms())])
        return values
    
    def save(self, path) :
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path) :
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    
class MoleculeEncodersTest(unittest.TestCase) :
    
    def setUp(self):
        self.smiles = ['CC(=O)NC1=CC=C(C=C1)O', 'OCl(=O)=O']
        self.mols = [Chem.MolFromSmiles(smiles) for smiles in self.smiles]
        self.cel = ConfEnsembleLibrary(mol_list=self.mols)
        self.molecule_encoders = MoleculeEncoders()
        #EmbedMultipleConfs(self.mol, 10) # conformations are not needed for encoding
    
    def test_get_all_function_values_in_library(self) :
        function = Atom.GetSymbol
        function_name = f'{function.__module__}.{function.__name__}'
        atomic_symbols = self.molecule_encoders.get_all_function_values_in_library(function_name=function_name,
                                                                                             conf_ensemble_library=self.cel)
        self.assertEqual(set(atomic_symbols), set(['C', 'N', 'O', 'Cl']))
    
    def test_create_one_hot_encoder(self):
        function = Atom.GetAtomicNum
        function_name = f'{function.__module__}.{function.__name__}'
        self.molecule_encoders.create_one_hot_encoder(function_name=function_name, 
                                                      conf_ensemble_library=self.cel)
        categories = self.molecule_encoders.get_encoder_categories(function_name)[0]
        atomic_symbols = self.molecule_encoders.get_all_function_values_in_library(function_name=function_name,
                                                                                   conf_ensemble_library=self.cel)
        self.assertEqual(len(categories), len(set(atomic_symbols)))
        
    def test_save_load(self) :
        function = Atom.GetAtomicNum
        function_name = f'{function.__module__}.{function.__name__}'
        self.molecule_encoders.create_one_hot_encoder(function_name=function_name, 
                                                      conf_ensemble_library=self.cel)
        test_path = 'test_mol_enc.p'
        self.molecule_encoders.save('test_mol_enc.p')
        loaded_molecule_encoders = MoleculeEncoders.load('test_mol_enc.p')
        self.assertEqual(self.molecule_encoders.encoders.keys(), loaded_molecule_encoders.encoders.keys())
        os.remove('test_mol_enc.p')
        
if __name__ == '__main__':
    unittest.main()