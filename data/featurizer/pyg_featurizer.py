import unittest
import torch

from typing import Sequence, List, Tuple, Dict
from .mol_featurizer import MolFeaturizer
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Chem.rdMolAlign import GetBestRMS
from rdkit.Chem import AllChem #needed for rdForceFieldHelpers
from rdkit.Chem.rdForceFieldHelpers import MMFFSanitizeMolecule
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from data.utils import MolConverter
from .molecule_encoders import MoleculeEncoders
from conf_ensemble import ConfEnsembleLibrary
from torch_geometric.data import Data
from ccdc.descriptors import MolecularDescriptors

    
class PyGFeaturizer(MolFeaturizer) :
    """
    Class to transform a molecule into a data point in torch geometric
    Inspired from the QM9 dataset from torch geometric
    """
    
    def __init__(self) :
        self.mol_converter = MolConverter()
        
    def featurize_mol(self, 
                      rdkit_mol: Mol, 
                      mol_ids: Sequence = None,
                      embed_hydrogens: bool = False) -> List[Data]:
        """
        Transforms all the conformations in the molecule into a list of torch
        geometric data
        
        :param rdkit_mol: Input molecule containing conformations to featurize
        :type rdkit_mol: Mol
        :param mol_ids: List of identifiers to give each conformation. Length
            must be the same as the number of conformations in the molecule
        :type mol_ids: Sequence
        :param embed_hydrogens: Whether to include the hydrogens in the data
        :type embed_hydrogens: bool
        :return: list of data, one for each conformation
        :rtype: List[Data]
        
        """
        
        if mol_ids :
            assert len(mol_ids) == rdkit_mol.GetNumConformers()
        
        data_list = []
        
        if not embed_hydrogens :
            rdkit_mol = Chem.RemoveHs(rdkit_mol)
        
        x = self.encode_atom_features(rdkit_mol)
        mol_bond_features, row, col = self.encode_bond_features(rdkit_mol)

        # Directed graph to undirected
        row, col = row + col, col + row
        edge_index = torch.tensor([row, col])
        edge_attr = torch.tensor(mol_bond_features + mol_bond_features, 
                                 dtype=torch.float32)

        # Sort the edge by source node idx
        perm = (edge_index[0] * rdkit_mol.GetNumAtoms() + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        
        # Make one data per conformer, because it has different positions
        confs = [conf for conf in rdkit_mol.GetConformers()]
        for i, conf in enumerate(confs) : 
            # i can be different than conf_id, i.e. if confs have been removed for a mol
            conf_id = conf.GetId()
            if mol_ids :
                mol_id = mol_ids[i]
            else :
                mol_id = Chem.MolToSmiles(rdkit_mol)
            data = self.conf_to_data(rdkit_mol=rdkit_mol, 
                                     conf_id=conf_id, 
                                     edge_index=edge_index, 
                                     x=x, 
                                     edge_attr=edge_attr,
                                     mol_id=mol_id)
            data_list.append(data)
            
        return data_list
    
    def encode_atom_features(self, 
                             rdkit_mol: Mol) -> torch.tensor :
        """
        Encode the atom features, here only the atomic number (can be modified)
        
        :param rdkit_mol: input molecule
        :type rdkit_mol: Mol
        :return: tensor (n_atoms, 1) storing the atomic numbers
        :rtype: torch.tensor
        
        """
        
        mol_atom_features = []
        for atom_idx, atom in enumerate(rdkit_mol.GetAtoms()):
            atom_features = []
            atom_features.append(atom.GetAtomicNum())
            mol_atom_features.append(atom_features)

        x = torch.tensor(mol_atom_features, dtype=torch.float32)
        return x
    
    def encode_bond_features(self, 
                             rdkit_mol: Mol) -> Tuple[list, List[int], List[int]] :
        """
        Encode the bond features, here none (can be modified)
        
        :param rdkit_mol: input molecule
        :type rdkit_mol: Mol
        :return: tuple storing an empty list, the list of starting atom in bonds
        and the list of ending atom in bonds.
        :rtype: Tuple[list, List[int], List[int]]
        
        """
        mol_bond_features = []
        row = []
        col = []
        for bond in rdkit_mol.GetBonds(): # bonds are undirect, while torch geometric data has directed edge
            bond_features = []
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row.append(start)
            col.append(end)

            mol_bond_features.append(bond_features)
        return mol_bond_features, row, col
    
    def conf_to_data(self, 
                     rdkit_mol: Mol, 
                     conf_id: int, 
                     edge_index: torch.tensor, 
                     x: torch.tensor = None, 
                     edge_attr: torch.tensor = None, 
                     save_mol: bool = False,
                     mol_id: str = None) -> Data: 
        """
        Create a torch geometric Data from a conformation
        
        :param rdkit_mol: input molecule
        :type rdkit_mol: Mol
        :param conf_id: id of the conformation to featurize in the molecule
        :type conf_id: int
        :param edge_index: tensor (n_bonds, 2) containing the start and end of
            each bond in the molecule
        :type edge_index: torch.tensor
        :param x: tensor containing the atomic numbers of each atom in the 
            molecule
        :type x: torch.tensor
        :param edge_attr: tensor to store other atom features (not used)
        :type edge_attr: torch.tensor
        :param save_mol: if True, will save the rdkit molecule as mol_id (not
            recommended, uses space)
        :type save_mol: bool
        :param mol_id: identifier of the conformation (saved in mol_id in Data)
        :type mol_id: str
        :return: single Data containing atomic numbers, positions and bonds for 
            the input conformation
        :rtype: Data
        
        """
        
        conf = rdkit_mol.GetConformer(conf_id)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        
        # keep only the embeded conformer
        # dummy_mol = copy.deepcopy(rdkit_mol)
        # dummy_mol.RemoveAllConformers()
        # dummy_mol.AddConformer(conf)
        
        # compute energy
        # dummy_mol = Chem.AddHs(dummy_mol, addCoords=True)
        
        # MMFFSanitizeMolecule(dummy_mol)
        # mol_properties = MMFFGetMoleculeProperties(dummy_mol)
        # mmff = MMFFGetMoleculeForceField(dummy_mol, mol_properties)
        # mmff_energy = mmff.CalcEnergy()

        # uff = AllChem.UFFGetMoleculeForceField(dummy_mol)
        # energy = uff.CalcEnergy()

        # y = torch.tensor(energy, requires_grad=False)
        # dummy_mol = Chem.RemoveHs(dummy_mol)
        
        # n_heavy_atoms = rdkit_mol.GetNumHeavyAtoms()
        # n_rotatable_bonds = CalcNumRotatableBonds(rdkit_mol)
        
        # if save_mol :
        #     data_id = dummy_mol
        # else :
        #     if mol_id :
        #         data_id = mol_id
        #     else :
        #         data_id = Chem.MolToSmiles(dummy_mol)
            
        data = Data(x=x, 
                    edge_index=edge_index, 
                    pos=pos,)
                    # data_id=data_id,
                    # energy=energy,
                    # n_heavy_atoms=n_heavy_atoms,
                    # n_rotatable_bonds=n_rotatable_bonds)
        
        return data
    
    def get_bioactive_rmsds(self, 
                            rdkit_mol: Mol, 
                            rmsd_func: str = 'ccdc') -> Dict[int, float]:
        """
        Compute the ARMSD to the closest bioactive conformation (they may be 
        more than one) for each conformation in the molecule.
        
        :param rdkit_mol: input molecule
        :type rdkit_mol: Mol
        :param rmsd_func: backend used to compute the rmsd_func. Whether ccdc
            (default) or rdkit
        :type rmsd_func: str
        :return: Dictionnary including the ARMSD for each confId
        :rtype: Dict[int, float]
        
        """
        
        bioactive_conf_ids = [conf.GetId() 
                              for conf in rdkit_mol.GetConformers() 
                              if not conf.HasProp('Generator')]
        assert len(bioactive_conf_ids) > 0, 'There is no bioactive conformation'
        
        rmsds = {}
        for conf in rdkit_mol.GetConformers() :
            conf_id = conf.GetId()

            if conf_id in bioactive_conf_ids :
                rmsd = 0
            else :
                rmsds_to_bioactive = []
                for bioactive_conf_id in bioactive_conf_ids :
                    if rmsd_func == 'ccdc' :
                        tested_ccdc_mol = self.mol_converter.rdkit_conf_to_ccdc_mol(rdkit_mol, conf_id)
                        bioactive_ccdc_mol = self.mol_converter.rdkit_conf_to_ccdc_mol(rdkit_mol, bioactive_conf_id)
                        rmsd = MolecularDescriptors.rmsd(tested_ccdc_mol, bioactive_ccdc_mol, overlay=True)
                    else :
                        rmsd = GetBestRMS(rdkit_mol, rdkit_mol, conf_id, bioactive_conf_id, maxMatches=1000)

                    rmsds_to_bioactive.append(rmsd)

                rmsd = min(rmsds_to_bioactive)
            
            rmsds[conf_id] = rmsd
            
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
        self.molecule_featurizer = PyGFeaturizer(molecule_encoders)
    
    def test_featurize_mol(self):
        data_list = self.molecule_featurizer.featurize_mol(self.mol)
        self.assertEqual(self.mol.GetNumConformers(), len(data_list))
        data_list = self.molecule_featurizer.featurize_mol(self.mol, 
                                                           embed_hydrogens=True)
        
    def test_get_bioactive_rmsds(self) :
        data_list = self.molecule_featurizer.featurize_mol(self.mol)
        rmsds = self.molecule_featurizer.get_bioactive_rmsds(data_list)
        for i, data in enumerate(data_list) :
            data.rmsd = rmsds[i]
        self.assertEqual(len(rmsds), len(data_list))
        
if __name__ == '__main__':
    unittest.main()