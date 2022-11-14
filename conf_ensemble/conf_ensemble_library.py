import os
import numpy as np
import pandas as pd

from typing import List, Dict
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm
from ccdc_rdkit_connector import CcdcRdkitConnector
from conf_ensemble import ConfEnsemble
from ccdc.descriptors import MolecularDescriptors
from molconfviewer import MolConfViewer

class ConfEnsembleLibrary() :
    """
    Class to store multiple molecules having each multiple confs. The backend
    is a dict having the ensemble names as keys.
    ConfEnsembleLibrary = CEL
    
    :param cel_name: name of the default directory to store ensembles in. In this
        work, it stores bioactive conformations from PDBBind
    :type cel_name: str
    :param root: data directory
    :type root: str
    
    """
    
    def __init__(self,
                 cel_name: str = 'pdb_conf_ensembles',
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data') -> None:
        self.root = root
        self.cel_name = cel_name
        self.cel_dir = os.path.join(self.root, self.cel_name)
        self.library = {}
        self.csv_filename = 'ensemble_names.csv'
        self.csv_filepath = os.path.join(self.cel_dir, self.csv_filename)
        self.cel_df = pd.DataFrame(columns=['ensemble_name', 'smiles', 'filename'])
        if not os.path.exists(self.cel_dir) :
            os.mkdir(self.cel_dir)
        else :
            self.load()
        
    
    @classmethod
    def from_mol_list(cls,
                      mol_list: List[Mol],
                      cel_name: str,
                      root: str,
                      names: List[str] = None) :
        """
        Constructor to create a library from a list of molecules (containing one
        or more confs).
        
        :param mol_list: list of input molecules
        :type mol_list: List[Mol]
        :param cel_name: name of the directory where the library will be stored
        :type cel_name: str
        :param root: data directory
        :type root: str
        :param names: list of ensemble name for each molecule, to be used to group
            in ensembles
        
        """
        
        conf_ensemble_library = cls(cel_name, root)
        if names is None :
            names = [Chem.MolToSmiles(mol) for mol in mol_list]
        else :
            assert len(mol_list) == len(names), \
                'mol_list and names should have the same length'
        
        for name, mol in zip(names, mol_list) :
            if not name in conf_ensemble_library.library :
                conf_ensemble_library.library[name] = ConfEnsemble(mol_list=[mol],
                                                                   name=name)
            else :
                conf_ensemble_library.library[name].add_mol(mol)
        return conf_ensemble_library
                
    @classmethod
    def from_mol_dict(cls,
                      mol_dict: Dict[str, Mol],
                      cel_name: str = 'pdb_conf_ensembles',
                      root: str = '/home/bb596/hdd/pdbbind_bioactive/data') :
        """
        Creates a ConfEnsemble from a dict of molecules each containing conformations
        
        :param mol_dict: dict of RDKit molecules
        :type mol_dict: Dict[str, Mol]
        :param cel_name: name of the library
        :type cel_name: str
        :param root: root directory of data storage
        :type root: str
        
        """
        conf_ensemble_library = cls(cel_name, root)
        for name, mol_list in tqdm(mol_dict.items()) :
            try :
                conf_ensemble_library.library[name] = ConfEnsemble(mol_list=mol_list,
                                                                   name=name)
                conf_ensemble_library.library[name].mol.SetProp('_Name', name)
            except Exception as e:
                print(f'conf ensemble failed for {name}')
                print(str(e))
        # conf_ensemble_library.save()
        return conf_ensemble_library
    
    
    @classmethod
    def from_ce_dict(cls,
                     ce_dict: Dict[str, ConfEnsemble],
                     cel_name: str = 'pdb_conf_ensembles',
                     root: str = '/home/bb596/hdd/pdbbind_bioactive/data'):
        """
        Creates a ConfEnsemble from a dict of conf ensembles
        
        :param mol_dict: dict of ConfEnsemble
        :type mol_dict: Dict[str, ConfEnsemble]
        :param cel_name: name of the library
        :type cel_name: str
        :param root: root directory of data storage
        :type root: str
        
        """
        conf_ensemble_library = cls(cel_name, root)
        for name, ce in tqdm(ce_dict.items()) :
            try :
                conf_ensemble_library.library[name] = ce
                conf_ensemble_library.library[name].mol.SetProp('_Name', name)
            except Exception as e:
                print(f'conf ensemble failed for {name}')
                print(str(e))
        # conf_ensemble_library.save()
        return conf_ensemble_library
    
            
    def load(self) :
        """
        Load the ConfEnsembles in the cel_dir given as input when creating the library
        and load associated metadata in cel_df
        
        """
        if os.path.exists(self.csv_filepath) :
            self.cel_df = pd.read_csv(self.csv_filepath)
            filenames = [f for f in os.listdir(self.cel_dir) if 'sdf' in f]
            print('Loading conf ensembles')
            for filename in tqdm(filenames) :
                filepath = os.path.join(self.cel_dir, filename)
                name = self.cel_df[self.cel_df['filename'] == filename]['ensemble_name'].values[0]
                try:
                    ce = ConfEnsemble.from_file(filepath, name)
                    self.library[name] = ce
                except:
                    print(f'Loading failed for {name}')
            
            
    def save(self) :
        """
        Save the ConfEnsembles in the cel_dir, and the metadata in cel_df and pdbbind_df
        
        """
        names = list(self.library.keys())
        print('Saving conf ensembles')
        for name_i, name in enumerate(tqdm(names)) :
            writer_filename = f'{name_i}.sdf'
            writer_path = os.path.join(self.cel_dir, writer_filename)
            ce = self.library[name]
            ce.save_ensemble(sd_writer_path=writer_path)
            smiles = Chem.MolToSmiles(ce.mol)
            row = pd.DataFrame([[name, smiles, writer_filename]], 
                               columns=self.cel_df.columns)
            self.cel_df = pd.concat([self.cel_df, row], ignore_index=True)
        self.cel_df.to_csv(self.csv_filepath, index=False)
        self.create_pdbbind_df()
            
            
    def merge(self,
              conf_ensemble_library: 'ConfEnsembleLibrary') :
        """
        Merge the ConfEnsembles from the input library and current library
        Only add conformations to molecules existing in the current library
        
        :param conf_ensemble_library: Input library to add confs from
        :type conf_ensemble_library: ConfEnsembleLibrary
        """
        for name in self.library :
            if name in conf_ensemble_library.library :
                ce = self.library[name]
                ce2 = conf_ensemble_library.library[name]
                try :
                    ce.add_mol(ce2.mol)
                except :
                    print(f"Merging didn't work for {name}")
            else :
                print(f'{name} is not in the second library')
                
                
    # we could use the PDBBindDataProcessor to do that
    def create_pdbbind_df(self) :
        """
        Create the DataFrame associating each ligand name to its pdb id
        for the current library
        """
        pdbbind_df = pd.DataFrame(columns=['ligand_name', 'pdb_id'])
        for name in self.library :
            mol = self.library[name].mol
            for conf in mol.GetConformers() :
                pdb_id = conf.GetProp('PDB_ID')
                row = pd.DataFrame([[name, pdb_id]], 
                                    columns=pdbbind_df.columns)
                pdbbind_df = pd.concat([pdbbind_df, row], ignore_index=True)
        pdbbind_df_path = os.path.join(self.root, 'pdbbind_df.csv')
        pdbbind_df.to_csv(pdbbind_df_path, index=False)
                
    @staticmethod
    def view_ensemble(name: str,
                      cel_name: str = 'gen_conf_ensembles',
                      root: str = '/home/bb596/hdd/pdbbind_bioactive/data'):
        """
        View the ConfEnsemble for the input molecule name from the 
        given cel_name using MolConfViewer
        
        :param name: Name of the molecule
        :type name: str
        :param cel_name: Name of the library to refer to
        :type cel_name: str
        :param root: Data directory
        :type root: str
        
        """
        cel_dir = os.path.join(root, cel_name)
        csv_filename = 'ensemble_names.csv'
        csv_filepath = os.path.join(cel_dir, csv_filename)
        cel_df = pd.read_csv(csv_filepath)
        try:
            filename = cel_df[cel_df['ensemble_name'] == name]['filename'].values[0]
            filepath = os.path.join(cel_dir, filename)
            ce = ConfEnsemble.from_file(filepath, name)
            MolConfViewer().view(ce.mol)
        except Exception as e:
            print(str(e))
            print(f'Loading failed for {name}')
            
    
    @classmethod   
    def get_merged_ce(cls,
                      filename: str, 
                      name: str,
                      root: str = '/home/bb596/hdd/pdbbind_bioactive/data',
                      cel_name1: str = 'pdb_conf_ensembles',
                      cel_name2: str = 'gen_conf_ensembles') -> ConfEnsemble: 
        """
        Merge the ConfEnsemble from 2 libraries for a molecule. We assume
        that the filename is the same for both libraries (e.g. bioactive and
        generated conformations of the same molecules in 2 different libraries)
        
        :param filename: Name of the file (i.sdf)
        :type filename: str
        :param name: Name of the molecule
        :type name: str
        :param root: Data directory
        :type root:str
        :param cel_name1: Name of the first library (e.g. containing bioactive conformations)
        :type cel_name1: str
        :param cel_name1: Name of the second library (e.g. containing generated conformations)
        :type cel_name1: str
        :param output: Type of the output
        :return: ConfEnsemble with conformations from the 2 libraries for the molecule
        :rtype: ConfEnsemble
        """
        cel_dir1 = os.path.join(root, cel_name1)
        
        ce_filepath = os.path.join(cel_dir1, filename)
        conf_ensemble = ConfEnsemble.from_file(filepath=ce_filepath, 
                                                name=name)
        
        cel_dir2 = os.path.join(root, cel_name2)
        gen_ce_filepath = os.path.join(cel_dir2, filename)
        gen_conf_ensemble = ConfEnsemble.from_file(filepath=gen_ce_filepath, 
                                                    name=name)
        
        gen_conf_ensemble.add_mol(conf_ensemble.mol, standardize=False)
        return gen_conf_ensemble