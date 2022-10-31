from typing import List
from rdkit import Chem
from rdkit.Chem import SDWriter
from rdkit.Chem.rdchem import Mol, Conformer
from mol_standardizer import MolStandardizer
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate

from pose_reader import PoseReader

class ConfEnsemble() :
    """
    Class able to store different confs for the same molecule in a single
    rdkit molecule. The workflow was made to standardize input molecules,
    check if they are the same, and add conformations with atom matching using
    rdkit molecule atom ordering
    
    :param mol_list: List of identical molecules but different confs
    :type mol_list: List[Mol]
    :param name: name of the ensemble. Default is template molecule smiles
    :type name: str
    :param template_mol: Molecule to serve as template, all new molecule to add
        will match this template (useful when you have a single 2D template i.e.
        PDB ligand expo)
    
    """
    
    def __init__(self,
                 mol_list: List[Mol],
                 name: str = None,
                 template_mol: Mol = None,
                 standardize: bool = True) -> None:
        self.name = name
        
        if template_mol is None:
            template_mol = mol_list[0]
            mol_list = mol_list[1:]
            
        if name is None:
            smiles = Chem.MolToSmiles(template_mol)
            name = smiles
        
        if standardize:
            self.mol_standardizer = MolStandardizer()
            template_mol = self.mol_standardizer.standardize(template_mol, neutralize=False)
        self.mol = template_mol
        
        for mol in mol_list :
            self.add_mol(mol, standardize)
            
        self.mol.SetProp('_Name', name)
            
                        
    def add_mol(self,
                mol: Mol,
                standardize: bool = True) :
        """
        Add molecule to the ensemble. Molecule must be the same as the template,
        only conf should be different
        
        :param mol: Molecule to add
        :type mol: Mol
        """
        
        if standardize:
            standard_mol = self.mol_standardizer.standardize(mol)
        else:
            standard_mol = mol
            
        match = standard_mol.GetSubstructMatch(self.mol)
        if len(match) != self.mol.GetNumHeavyAtoms() :
            # Try using bond order assignement
            standard_mol = AssignBondOrdersFromTemplate(self.mol, standard_mol)
            match = standard_mol.GetSubstructMatch(self.mol)
            if not len(match) == self.mol.GetNumAtoms() :
                raise Exception('No match found between template and actual mol')
            
        renumbered_mol = Chem.RenumberAtoms(standard_mol, match)
        for new_conf in renumbered_mol.GetConformers() :
            conf_id = new_conf.GetId()
            original_conf = standard_mol.GetConformer(conf_id)
            prop_names = original_conf.GetPropNames(includePrivate=True, 
                                                    includeComputed=True)
            for prop in prop_names:
                value = original_conf.GetProp(prop)
                new_conf.SetProp(prop, str(value))
            self.mol.AddConformer(new_conf, assignId=True)
        
    
    @classmethod
    def from_file(cls,
                  filepath: str,
                  name: str = None,
                  output: str = 'conf_ensemble',
                  standardize: bool = False) :
        """
        Constructor to create a conf ensemble from a sdf file
        
        :param filepath: path to the sdf file containing multiple confs of the
        same molecule
        :type filepath: str
        :param name: name of the ensemble
        :type name: str
        """
        
        if filepath.endswith('.sdf'):
            with open(filepath, 'rb') as f:
                suppl = Chem.ForwardSDMolSupplier(f)
                mol_list = [mol for mol in suppl]
        else:
            suppl = PoseReader(filepath)
            mol_list = [mol for mol in suppl]
            
        def mol_props_to_conf(mol) :
            prop_names = mol.GetPropNames(includePrivate=True, 
                                         includeComputed=True)
            for prop in prop_names :
                value = mol.GetProp(prop)
                for conf in mol.GetConformers() :
                    conf.SetProp(prop, str(value))
                mol.ClearProp(prop)
                
        for mol in mol_list :
            mol_props_to_conf(mol)
            
        if name is None :
            name = Chem.MolToSmiles(mol_list[0])
            
        if output == 'conf_ensemble':
            ce = cls(mol_list, name, standardize=standardize)
            return ce
        else:
            return mol_list
            
            
    def save_ensemble(self,
                      sd_writer_path: str) :
        """
        Save the ensemble in an SDF file
        
        :param sd_writer_path: SDF file path to store all confs for the molecule
        :type sd_writer_path: str
            
        """
        
        sd_writer = SDWriter(sd_writer_path)
        self.save_confs_to_writer(writer=sd_writer)
        
        
    def save_confs_to_writer(self,
                             writer: SDWriter) :
        """
        Save each conf of the RDKit mol as a single molecule in a SDF
        
        :param writer: RDKit writer object
        :type writer: SDWriter
        
        """
        mol = Mol(self.mol)
        for conf in mol.GetConformers() :
            conf_id = conf.GetId()
            # Store the conf properties as molecule properties to save them
            prop_names = conf.GetPropNames(includePrivate=True, 
                                           includeComputed=True)
            for prop in prop_names :
                value = conf.GetProp(prop)
                mol.SetProp(prop, str(value))
            writer.write(mol, conf_id)