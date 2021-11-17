import pickle
import torch
import os
import copy

from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import to_undirected
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, RingInfo
from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
from typing import List
from ConformationEnsemble import ConformationEnsembleLibrary, ConformationEnsemble
from tqdm import tqdm
from rdkit.Chem.rdMolAlign import GetBestRMS
from ccdc.conformer import ConformerGenerator
from ccdc.molecule import Molecule
from ccdc.descriptors import MolecularDescriptors
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem.TorsionFingerprints import GetTFDBetweenConformers

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

class OneHotEncoder(object) :
        
    def __init__(self, classes: List) :
        self.classes = classes

    def __call__(self, label) :
        return [1 if label == c else 0 for c in self.classes]

class MoleculeEncoders(object) :
    
    """
    Build one hot encoders for molecule feature extraction functions, based on available values in a 
    conformation ensemble library. A dict is used to store OneHotEncoder objects using the function name
    (because RDKit functions are from Boost, and therefore not picklable)
    """
    
    def __init__(self) :
        self.encoders = {}
    
    def create_one_hot_encoder(self, function_name, conf_ensemble_library: ConformationEnsembleLibrary) :
        function = eval(function_name)
        values_set = set()
        for smiles, conf_ensemble in conf_ensemble_library.get_unique_molecules() :
            if function.__module__ == 'Atom' : 
                values = [function(atom) for atom in conf_ensemble.mol.GetAtoms()]
            elif function.__module__ == 'Bond' :
                values = [function(bond) for bond in conf_ensemble.mol.GetBonds()]
            elif function.__module__ == 'RingInfo' :
                ring_info = conf_ensemble.mol.GetRingInfo()
                values = [function(ring_info, atom_idx) for atom_idx, atom in enumerate(conf_ensemble.mol.GetAtoms())]
            values_set.update(values)
        self.encoders[function_name] = OneHotEncoder(sorted(list(values_set)))
    
    def __getitem__(self, function_name) :
        assert function_name in self.encoders, 'The function you are asking is not encoded, please use create_one_hot_encoder'
        return self.encoders[function_name]
    
    def get_encoder_classes(self, function_name) :
        return self.encoders[function_name].classes


class ConfEnsembleDataset(InMemoryDataset) :
    
    def __init__(self, root, encoders: MoleculeEncoders=None, exclude_hydrogens=True):
        
        self.root = root
        
        self.ccdc_conformer_generator = ConformerGenerator()
        self.exclude_hydrogens = exclude_hydrogens
        
        # Build encoders depending on existing features in the dataset
        self.encoded_atom_functions = [Atom.GetAtomicNum, Atom.GetDegree, Atom.GetHybridization, 
                                       Atom.GetChiralTag, Atom.GetImplicitValence, Atom.GetFormalCharge]
        self.encoded_bond_functions = [Bond.GetBondType]
        self.encoded_ring_functions = [RingInfo.NumAtomRings]
        
        self.pdbbind_encoder_path = os.path.join(self.root, 'pdbbind_molecule_encoders.p')
        
        with open(self.raw_paths[0], 'rb') as f:
            conf_ensemble_library = pickle.load(f)
        
        if os.path.exists(self.pdbbind_encoder_path) : # Load existing
            with open(self.pdbbind_encoder_path, 'rb') as f:
                self.encoders = pickle.load(f)
        else : # Create encoders
            self.encoders = MoleculeEncoders()
            print('Creating molecule encoders')
            for function in tqdm(self.encoded_atom_functions + self.encoded_bond_functions + self.encoded_ring_functions) :
                function_name = f'{function.__module__}.{function.__name__}'
                self.encoders.create_one_hot_encoder(function_name, conf_ensemble_library)
            with open(self.pdbbind_encoder_path, 'wb') as f:
                pickle.dump(self.encoders, f)
        
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def raw_file_names(self) -> List[str]:
        return ['pdbbind_conf_ensemble_library.p']
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
    
    def process(self):
        
        with open(self.raw_paths[0], 'rb') as f:
            conf_ensemble_library = pickle.load(f)
        
        master_data_list = []
        #for smiles, conf_ensemble in tqdm(conf_ensemble_library.get_unique_molecules()) :
        for smiles, conf_ensemble in tqdm(list(conf_ensemble_library.get_unique_molecules())[:1000]) :
            try :
                mol = conf_ensemble.mol
                data_list = self.featurize_mol(mol)
                master_data_list.extend(data_list)
            except Exception as e : 
                print('Error for smiles :' + smiles)
                print(e)
        
        torch.save(self.collate(master_data_list), self.processed_paths[0])
        
    def featurize_mol(self, mol, conf_generator='ccdc', rmsd_func='ccdc') :
        
        data_list = []
        
        #import pdb; pdb.set_trace()
        
        # Generate conformations
        n_conf = mol.GetNumConformers()
        bioactive_conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        
        #mol_with_decoys = copy.deepcopy(mol)
        n_conf_to_generate = 50 - n_conf
        if conf_generator == 'rdkit' :
            rdkit_conf_ids = EmbedMultipleConfs(mol, n_conf_to_generate, clearConfs=False)
        elif conf_generator == 'ccdc' :
            mol2_block = Chem.MolToMolBlock(mol)
            
            ccdc_mol = Molecule.from_string(mol2_block)
            self.ccdc_conformer_generator.settings.max_conformers = n_conf_to_generate
            ccdc_conformers = self.ccdc_conformer_generator.generate(ccdc_mol)
            
            self.ccdc_conformers_to_rdkit_mol(mol, ccdc_conformers, exclude_hydrogens=self.exclude_hydrogens)
        
        if self.exclude_hydrogens :
            mol = Chem.RemoveHs(mol)
        
        # Encode atom features
        mol_atom_features = []
        ring = mol.GetRingInfo()
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            atom_features = []

            # Add one hot encoding of atomic features
            for function in self.encoded_atom_functions :
                function_name = f'{function.__module__}.{function.__name__}'
                encoder = self.encoders[function_name]
                value = function(atom)
                atom_features.extend(encoder(value))

            atom_features.append(atom.GetAtomicNum()) # Atomic number is both encoded as one hot and integer
            atom_features.append(1 if atom.GetIsAromatic() else 0)

            atom_features.extend([atom.IsInRingSize(size) for size in range(3, 9)])

            # Add one hot encoding of ring features in atomic features
            for function in self.encoded_ring_functions :
                function_name = f'{function.__module__}.{function.__name__}'
                encoder = self.encoders[function_name]
                value = function(ring, atom_idx)
                atom_features.extend(encoder(value))

            mol_atom_features.append(atom_features)

        x = torch.tensor(mol_atom_features, dtype=torch.float32)

        # Encode bond features
        mol_bond_features = []
        row = []
        col = []
        for bond in mol.GetBonds(): # bonds are undirect, while torch geometric data has directed edge
            bond_features = []
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row.append(start)
            col.append(end)

            # Add one hot encoding of bond features
            for function in self.encoded_bond_functions :
                function_name = f'{function.__module__}.{function.__name__}'
                encoder = self.encoders[function_name]
                value = function(bond)
                bond_features.extend(encoder(value))

            bond_features.append(int(bond.IsInRing()))
            bond_features.append(int(bond.GetIsConjugated()))
            bond_features.append(int(bond.GetIsAromatic()))

            mol_bond_features.append(bond_features)

        # Directed graph to undirected
        row, col = row + col, col + row
        edge_index = torch.tensor([row, col])
        edge_attr = torch.tensor(mol_bond_features + mol_bond_features, dtype=torch.float32)

        # Sort the edge by source node idx
        perm = (edge_index[0] * mol.GetNumAtoms() + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        
        # Make one data per conformer, because it has different positions
        for conf_id, conf in enumerate(mol.GetConformers()) :
            
            # Keep only the conformer encoded in by the positions
            dummy_mol = copy.deepcopy(mol)
            dummy_mol.RemoveAllConformers()
            dummy_mol.AddConformer(mol.GetConformer(conf_id))
            
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
            
            if conf_id in bioactive_conf_ids :
                rmsd = 1e-6
                tfd = 1e-6
            else :
                rmsds_to_bioactive = []
                tfds_to_bioactive = []
                for bioactive_conf_id in bioactive_conf_ids :
                    
                    if rmsd_func == 'ccdc' :
                        tested_molblock = Chem.MolToMolBlock(mol, confId=conf_id)
                        bioactive_molblock = Chem.MolToMolBlock(mol, confId=bioactive_conf_id)
                        tested_ccdc_mol = Molecule.from_string(tested_molblock)
                        bioactive_ccdc_mol = Molecule.from_string(bioactive_molblock)
                        rmsd = MolecularDescriptors.overlay_rmsds_and_transformation(tested_ccdc_mol, bioactive_ccdc_mol)[1]
                    else :
                        rmsd = GetBestRMS(mol, mol, conf_id, bioactive_conf_id, maxMatches=1000)
                        
                    rmsds_to_bioactive.append(rmsd)
                    
                    tfd = GetTFDBetweenConformers(mol, [conf_id], [bioactive_conf_id])[0]
                    tfds_to_bioactive.append(tfd)
                    
                rmsd = min(rmsds_to_bioactive)
                tfd = min(tfds_to_bioactive)
            
            if tfd > 1 : # weird molecule
                print(tfd)
                print(Chem.MolToSmiles(dummy_mol))
                tfd = 1
            
            #y = 1 if ('PDB_ID' in conf.GetPropsAsDict()) else 0
            rmsd = torch.tensor([rmsd], dtype=torch.float32)
            tfd = torch.tensor([tfd], dtype=torch.float32)
            
            #data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, mol=dummy_mol, y=y)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, rmsd=rmsd, tfd=tfd, smiles=Chem.MolToSmiles(dummy_mol))
            data_list.append(data)
            
        return data_list

    def ccdc_conformers_to_rdkit_mol(self, rdkit_mol, ccdc_conformers, exclude_hydrogens=True) :
        for conformer in ccdc_conformers :
            temp_rdkit_mol = copy.deepcopy(rdkit_mol)
            new_rdkit_conf = temp_rdkit_mol.GetConformer()
            conformer_ccdc_molecule = conformer.molecule
#             if exclude_hydrogens :
#                 conformer_ccdc_molecule.remove_hydrogens()
            for i, atom in enumerate(conformer_ccdc_molecule.atoms) :
                atom_coord = [coord for coord in atom.coordinates]
                point3d = Point3D(*atom_coord)
                new_rdkit_conf.SetAtomPosition(i, point3d)
            rdkit_mol.AddConformer(new_rdkit_conf, assignId=True)
            