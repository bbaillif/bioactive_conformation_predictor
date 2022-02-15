import unittest
import torch
import copy

from rdkit import Chem
from rdkit.Chem.rdchem import (
    Atom, 
    Bond, 
    RingInfo, 
    Mol)
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Chem.rdMolAlign import GetBestRMS
from rdkit.Chem import AllChem #needed for rdForceFieldHelpers
from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeProperties, MMFFGetMoleculeForceField, MMFFSanitizeMolecule
from ccdc_rdkit_connector import CcdcRdkitConnector
from molecule_encoders import MoleculeEncoders
from conf_ensemble import ConfEnsembleLibrary
from torch_geometric.data import Data
from ccdc.descriptors import MolecularDescriptors
    
class MoleculeFeaturizer() :
    
    def __init__(self, encoders, encode_graph=False) :
        self.molecule_encoders = encoders
        self.encoded_atom_function_names = encoders.encoded_atom_function_names
        self.encoded_bond_function_names = encoders.encoded_bond_function_names
        self.encoded_ring_function_names = encoders.encoded_ring_function_names
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
        self.encode_graph = encode_graph
        
    def featurize_mol(self, rdkit_mol, conf_generator='ccdc', rmsd_func='ccdc', interpolate=False, exclude_hydrogens=True) :
        
        data_list = []
        
        if exclude_hydrogens :
            rdkit_mol = Chem.RemoveHs(rdkit_mol)
        
        x = self.encode_atom_features(rdkit_mol)
        mol_bond_features, row, col = self.encode_bond_features(rdkit_mol)

        # Directed graph to undirected
        row, col = row + col, col + row
        edge_index = torch.tensor([row, col])
        edge_attr = torch.tensor(mol_bond_features + mol_bond_features, dtype=torch.float32)

        # Sort the edge by source node idx
        perm = (edge_index[0] * rdkit_mol.GetNumAtoms() + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        
        # Make one data per conformer, because it has different positions
        for conf_id, conf in enumerate(rdkit_mol.GetConformers()) :
            data = self.conf_to_data(rdkit_mol, conf_id, x, edge_index, edge_attr)
            data_list.append(data)
            
        return data_list
    
    def encode_atom_features(self, rdkit_mol) :
        
        mol_atom_features = []
        ring = rdkit_mol.GetRingInfo()
        for atom_idx, atom in enumerate(rdkit_mol.GetAtoms()):
            atom_features = []

            atom_features.append(atom.GetAtomicNum()) # Atomic number is both encoded as one hot and integer
            
            # Add one hot encoding of atomic features
            for function_name in self.encoded_atom_function_names :
                function = eval(function_name)
                one_hot_encoder = self.molecule_encoders[function_name]
                value = function(atom)
                atom_features.extend(one_hot_encoder.transform([[value]])[0])
            
            atom_features.append(1 if atom.GetIsAromatic() else 0)

            atom_features.extend([atom.IsInRingSize(size) for size in range(3, 9)])

            # Add one hot encoding of ring features in atomic features
            for function_name in self.encoded_ring_function_names :
                function = eval(function_name)
                one_hot_encoder = self.molecule_encoders[function_name]
                value = function(ring, atom_idx)
                atom_features.extend(one_hot_encoder.transform([[value]])[0])

            mol_atom_features.append(atom_features)

        x = torch.tensor(mol_atom_features, dtype=torch.float32)
        return x
    
    def encode_bond_features(self, rdkit_mol) :
        mol_bond_features = []
        row = []
        col = []
        for bond in rdkit_mol.GetBonds(): # bonds are undirect, while torch geometric data has directed edge
            bond_features = []
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row.append(start)
            col.append(end)

            # Add one hot encoding of bond features
            for function_name in self.encoded_bond_function_names :
                function = eval(function_name)
                one_hot_encoder = self.molecule_encoders[function_name]
                value = function(bond)
                bond_features.extend(one_hot_encoder.transform([[value]])[0])

            bond_features.append(int(bond.IsInRing()))
            bond_features.append(int(bond.GetIsConjugated()))
            bond_features.append(int(bond.GetIsAromatic()))

            mol_bond_features.append(bond_features)
        return mol_bond_features, row, col
    
    def conf_to_data(self, rdkit_mol, conf_id, x, edge_index, edge_attr, save_mol=False) :
        
        conf = rdkit_mol.GetConformer(conf_id)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        
        # keep only the embeded conformer
        dummy_mol = copy.deepcopy(rdkit_mol)
        dummy_mol.RemoveAllConformers()
        dummy_mol.AddConformer(conf)
        
        # compute energy
        dummy_mol = Chem.AddHs(dummy_mol, addCoords=True)
        
        MMFFSanitizeMolecule(dummy_mol)
        mol_properties = MMFFGetMoleculeProperties(dummy_mol)
        mmff = MMFFGetMoleculeForceField(dummy_mol, mol_properties)
        mmff_energy = mmff.CalcEnergy()

        uff = AllChem.UFFGetMoleculeForceField(dummy_mol)
        energy = uff.CalcEnergy()

        y = torch.tensor(energy, requires_grad=False)
        dummy_mol = Chem.RemoveHs(dummy_mol)
        
        if save_mol :
            data_id = dummy_mol
        else :
            data_id = Chem.MolToSmiles(dummy_mol)
            
        if self.encode_graph :
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, data_id=data_id, energy=energy)
        else :
            z = x[:, 0]
            data = Data(z=z, edge_index=edge_index, pos=pos, data_id=data_id, energy=energy)
            
        return data
    
    def get_bioactive_rmsds(self, rdkit_mol, rmsd_func='ccdc') :
        
        bioactive_conf_ids = [conf.GetId() for conf in rdkit_mol.GetConformers() if not conf.HasProp('Generator')]
        generated_conf_ids = [conf.GetId() for conf in rdkit_mol.GetConformers() if conf.HasProp('Generator')]
        
        rmsds = []
        for conf in rdkit_mol.GetConformers() :
            conf_id = conf.GetId()

            if conf_id in bioactive_conf_ids :
                rmsd = 0
            else :
                rmsds_to_bioactive = []
                for bioactive_conf_id in bioactive_conf_ids :
                    if rmsd_func == 'ccdc' :
                        tested_ccdc_mol = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol, conf_id)
                        bioactive_ccdc_mol = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol, bioactive_conf_id)
                        rmsd = MolecularDescriptors.rmsd(tested_ccdc_mol, bioactive_ccdc_mol, overlay=True)
                    else :
                        rmsd = GetBestRMS(rdkit_mol, rdkit_mol, conf_id, bioactive_conf_id, maxMatches=1000)

                    rmsds_to_bioactive.append(rmsd)

                rmsd = min(rmsds_to_bioactive)
            
            rmsds.append(rmsd)
            
        rmsds = torch.tensor(rmsds, dtype=torch.float32)
            
        return rmsds
        
        
class MoleculeFeaturizerTest(unittest.TestCase):

    def setUp(self):
        
        self.smiles = 'CC(=O)NC1=CC=C(C=C1)O'
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.mol = Chem.AddHs(self.mol)
        EmbedMultipleConfs(self.mol, 10)
        for conf in self.mol.GetConformers()[5:] :
            conf.SetProp('Generator', 'CCDC')
        self.cel = ConfEnsembleLibrary([self.mol])
        
        molecule_encoders = MoleculeEncoders()
        molecule_encoders.create_encoders(self.cel)
        self.molecule_featurizer = MoleculeFeaturizer(molecule_encoders)
    
    def test_featurize_mol(self):
        data_list = self.molecule_featurizer.featurize_mol(self.mol)
        self.assertEqual(self.mol.GetNumConformers(), len(data_list))
        data_list = self.molecule_featurizer.featurize_mol(self.mol, exclude_hydrogens=False)
        
    def test_get_bioactive_rmsds(self) :
        data_list = self.molecule_featurizer.featurize_mol(self.mol)
        rmsds = self.molecule_featurizer.get_bioactive_rmsds(data_list)
        for i, data in enumerate(data_list) :
            data.rmsd = rmsds[i]
        self.assertEqual(len(rmsds), len(data_list))
        
if __name__ == '__main__':
    unittest.main()