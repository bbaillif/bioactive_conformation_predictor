import os
import pickle
import copy
import unittest
import pandas as pd

from typing import List, Dict, Tuple, Union
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
from rdkit.Chem.rdmolfiles import SDWriter, SDMolSupplier
from ccdc_rdkit_connector import CcdcRdkitConnector
from ccdc.conformer import ConformerGenerator, ConformerHitList
from ccdc.io import Molecule
from multiprocessing import Pool
from tqdm import tqdm
from conf_ensemble import ConfEnsemble

# Could be more pythonic using MutableMapping ABC
class ConfEnsembleLibrary() :
    """
    Class to store multiple ConfEnsemble in a dict where the 
    smiles is the key.
    The default behaviour is to create library of bioactive conformations,
    then to generate conformations for these, and have a final library
    containing bio+gen conformations
    Args :
        root: str = data directory to store the conf ensembles
        conf_ensemble_dir: str = directory to store the bioactive conformer
            ensemble
        conf_ensemble_df_name: str = filename for the metadata csv file
        gen_conf_ensemble_dir: str = directory to store the generated
            conformers of molecules in the library (after generation)
        merged_conf_ensemble_dir: str = directory to store the bioactive +
            generated conf ensembles
    """

    def __init__(self, 
                 root: str='/home/bb596/hdd/pdbbind_bioactive/data/',
                 conf_ensemble_dir: str='conf_ensembles/',
                 conf_ensemble_df_name: str='index.csv',
                 gen_conf_ensemble_dir: str='gen_conf_ensembles/',
                 merged_conf_ensemble_dir: str='all_conf_ensembles/') :
        self.root = root
        self.conf_ensemble_dir = os.path.join(self.root, conf_ensemble_dir)
        self.conf_ensemble_df_name = conf_ensemble_df_name
        self.gen_conf_ensemble_dir = os.path.join(self.root, gen_conf_ensemble_dir)
        self.merged_conf_ensemble_dir = os.path.join(self.root, merged_conf_ensemble_dir)
        
        if not os.path.exists(self.conf_ensemble_dir) :
            os.mkdir(self.conf_ensemble_dir)
        if not os.path.exists(self.gen_conf_ensemble_dir) :
            os.mkdir(self.gen_conf_ensemble_dir)
        if not os.path.exists(self.merged_conf_ensemble_dir) :
            os.mkdir(self.merged_conf_ensemble_dir)
        self.conf_ensemble_df_path = os.path.join(self.conf_ensemble_dir,
                                                  self.conf_ensemble_df_name)
        
        self.table = pd.DataFrame(columns=['smiles', 'file_name'])
        self.library = {}
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
         
    @classmethod
    def from_mol_list(cls,
                      mol_list: List[Mol],
                      standardize: bool = True) :
        """
        Creates a library from a list of molecules
        Args:
            mol_list: List[Mol] = input molecules
            standardize_mols: bool = whether to standardize the mol when
                creating the conf ensembles
        """
        conf_ensemble_library = cls()
        print('Generating library')
        for mol in tqdm(mol_list) :
            smiles = Chem.MolToSmiles(mol)
            try :
                if smiles in conf_ensemble_library.library :
                    conf_ensemble_library.library[smiles].add_confs_from_mol(mol,
                                                                             standardize=standardize)
                else :
                    conf_ensemble_library.library[smiles] = ConfEnsemble(mol=mol,
                                                                         standardize_mols=standardize)
            except Exception as e :
                print('Molecule failed')
                print(str(e))
                
        return conf_ensemble_library
    
    
    @classmethod
    def from_dirs(cls,
                  dir_pathes: List[str]) :
        """
        Creates a library from directories containing sdf files
        Args: dir_pathes: List[str] = list of directories containing sdf files
        
        """
        print('Generating library')
        mol_list = []
        for dir_path in dir_pathes:
            file_names = os.listdir(dir_path)
            for file_name in file_names :
                assert file_name.endswith('.sdf')
                file_path = os.path.join(dir_path, file_name)
                mol_list.extend([mol for mol in SDMolSupplier(file_path)])
        return cls.from_mol_list(mol_list=mol_list)
              
    @classmethod  
    def from_pickle(cls,
                    file_path: str) :
        """
        Creates a library from a pickle file storing a ConfEnsembleLibrary
        Args:
            file_path: str = path to the pickle file
        """
        with open(file_path, 'rb') as f :
            cel = pickle.load(f) 
        return cel
                
                
    def load(self, 
             library_name: str = 'bioactive') :
        """
        Load a existing library from one of the default directory
        Args:
            library_name: str = name of the library
        """
        if library_name == 'bioactive' :
            load_dir = self.conf_ensemble_dir
        elif library_name == 'merged' :
            load_dir = self.merged_conf_ensemble_dir
        elif library_name == 'generated' :
            load_dir = self.gen_conf_ensemble_dir
        self.load_metadata()
            
        for smiles, file_name in tqdm(self.table.values):
            try :
                file_path = os.path.join(load_dir, file_name)
                conf_ensemble = self.get_ensemble_from_file(file_path=file_path)
                self.library[smiles] = conf_ensemble
            except Exception as e :
                print('Error with ' + file_path)
                print(str(e))
               
               
    def load_metadata(self) :
        """
        Load default metatada (store in self.conf_ensemble_df_path)
        """
        self.table = pd.read_csv(self.conf_ensemble_df_path, index_col=0)
               
               
    def get_ensemble_from_smiles(self,
                                 smiles: str,
                                 library_name: str = 'bioactive'
                                 ) -> ConfEnsemble:
        """
        Get ConfEnsemble for a smiles
        Args:
            smiles: str = SMILES to load
            library_name: str = name of the library to extract the smiles from
        Returns:
            ConfEnsemble = ensemble for the SMILES
        """
        if library_name is None or library_name == 'bioactive' :
            load_dir = self.conf_ensemble_dir
        elif library_name == 'generated' :
            load_dir = self.gen_conf_ensemble_dir
        elif library_name == 'merged' :
            load_dir = self.merged_conf_ensemble_dir
            
        smiles_table = self.table[self.table['smiles'] == smiles]
        if len(smiles_table) :
            file_name = smiles_table['file_name'].values[0]
            file_path = os.path.join(load_dir,
                                     file_name)
            return self.get_ensemble_from_file(file_path=file_path)
        else :
            raise Exception('Smiles not in conf ensemble')
               
               
    def get_ensemble_from_file(self, 
                               file_path: str) -> ConfEnsemble:
        """
        Get the ConfEnsemble from a sdf file
        Args:
            file_path: str = SDF file to a standardized conf ensemble
        Returns:
            ConfEnsemble = ensemble for the molecule in the SDF
        """
        try :
            assert file_path.endswith('.sdf')
            suppl = SDMolSupplier(file_path)
            return ConfEnsemble(mol_list=suppl,
                                standardize_mols=False)
        except Exception as e:
            print(str(e))
            raise Exception(f'{file_path} does not exist')
               
                
    def save(self, 
             library_name: str = 'bioactive',
             save_table: bool = True,
             unique_filename: str = None) :
        """
        Save the library in SDF files in a directory by default; if 
        unique_filename is given, then store in a single SDF
        Args:
            library_name: str = name of the library to save the data
            save_table: bool = save the metadata in a CSV file
            unique_filename: str = filename if the user decides to save the ensemble
                in a single sdf file
        """
        if library_name == 'bioactive' :
            save_dir = self.conf_ensemble_dir
        elif library_name == 'merged' :
            save_dir = self.merged_conf_ensemble_dir
        
        if unique_filename :
            unique_filepath = os.path.join(self.root, unique_filename)
            writer = SDWriter(unique_filepath)
        
        i = 0
        for smiles, conf_ensemble in tqdm(self.library.items()):
            mol = conf_ensemble.mol
            file_name = f'{i}.sdf'
            if unique_filepath :
                self.save_confs_to_writer(rdkit_mol=mol,
                                          writer=writer)
            else :
                writer_path = os.path.join(save_dir,
                                       file_name)
                self.save_ensemble(rdkit_mol=mol,
                                   sd_writer_path=writer_path)
            row = pd.DataFrame([[smiles, file_name]], 
                               columns=self.table.columns)
            self.table = pd.concat([self.table, row], ignore_index=True)
            i = i + 1
            
        if unique_filepath :
            writer.close()

        if save_table :
            self.table.to_csv(self.conf_ensemble_df_path)


    def save_ensemble(self,
                      rdkit_mol: Mol,
                      sd_writer_path: str) :
        """
        Save an ensemble (RDKit Mol with multiple confs) in given filepath
        Args:
            rdkit_mol: Mol = RDKit molecule containing multiple confs
            sd_writer_path: str = SDF file path to store all confs for the molecule
        """
        with SDWriter(sd_writer_path) as sd_writer :
            self.save_confs_to_writer(rdkit_mol=rdkit_mol,
                                      writer=sd_writer)


    def save_confs_to_writer(self,
                             rdkit_mol: Mol,
                             writer: SDWriter) :
        """
        Save each conf of a RDKit mol as a single molecule in a SDF
        Args:
            rdkit_mol: Mol = input molecule
            writer: SDWriter = to store each conf
        """
        mol = copy.deepcopy(rdkit_mol)
        for conf in mol.GetConformers() :
            conf_id = conf.GetId()
            # Store the conf properties as molecule properties to save them
            for prop in conf.GetPropNames() :
                value = conf.GetProp(prop)
                mol.SetProp(prop, str(value))
            writer.write(mol, conf_id)


    def get_num_molecules(self) -> int:
        """
        Get the number of unique molecules in the library
        Returns:
            int = number of molecules
        """
        return len(self.library)
    
    
    def get_conf_ensemble(self, 
                          smiles: str, 
                          canonical_check: bool = True) -> ConfEnsemble:
        """
        Get the conf ensemble for a given SMILES
        Args:
            smiles: str = input SMILES
            canonical_check: bool = if True, will look for the canonical
                smiles in the library
        Returns:
            ConfEnsemble = conf ensemble for the input SMILES
        """
        if canonical_check :
            canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            assert canon_smiles in self.library, \
            'Canonical input smiles not in library'
        return self.library[smiles]
    
    
    def get_unique_molecules(self) -> Dict[str, ConfEnsemble]:
        """
        Return the library as a dict
        Returns :
            Dict[str, ConfEnsemble] : mapping SMILES to corresponding conf 
                ensemble
        """
        return self.library.items()
    
    
    def merge(self, 
              second_library: 'ConfEnsembleLibrary',
              inplace: bool = False) -> Union['ConfEnsembleLibrary', None]:
        """
        Merge the current library with another input library
        Args:
            second_library: ConfEnsembleLibrary = second library to add
                confs from
            inplace: bool = if True, add the confs into current and return None,
                else returns a library
        Returns:
            Union['ConfEnsembleLibrary', None]: None if inplace, else new 
                library 
        """
        if not inplace :
            conf_ensemble_library = copy.deepcopy(self)
        else :
            conf_ensemble_library = self
            
        for smiles, conf_ensemble in second_library.get_unique_molecules() :
            mol = conf_ensemble.mol
            if smiles in self.library :
                conf_ensemble_library.library[smiles].add_confs_from_mol(mol)
            else :
                conf_ensemble_library.library[smiles] = ConfEnsemble(mol=mol)
                
        if not inplace :
            return conf_ensemble_library
        else :
            return None
    
    
    def remove_smiles(self, 
                      smiles: str) :
        """
        Remove smiles from library
        Args:
            smiles: str = SMILES to remove from library
        """
        if smiles in self.library :
            self.library.pop(smiles)
        else :
            print(f'Input smiles {smiles} not in library')
        
        
    def generate_conf_pool(self, 
                           included_smiles: list) :
        """
        Uses multiprocessing to generate confs for a list of smiles in the 
        library
        Args:
            included_smiles : list = SMILES to generate confs for
        """
        params = []
    
        included_table = self.table[self.table['smiles'].isin(included_smiles)]
        for i, row in included_table.iterrows() :
            smiles = row['smiles']
            file_name = row['file_name']
            params.append((smiles, file_name))
            
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.generate_conf_thread, params)
            
    def generate_conf_thread(self, 
                             params: Tuple[str, str]) :
        """
        Thread for multiprocessing Pool to generate confs for a molecule using
        the CCDC conformer generator. Default behaviour is to save all
        generated confs in a dir, and initial+generated ensemble in a 'merged'
        dir
        Args:
            params: Tuple[str, str] = SMILES and SDF file name of the molecule
                to generate conformations for
        """
        
        smiles, file_name = params
        original_file_path = os.path.join(self.conf_ensemble_dir,
                                          file_name)
        try :
            gen_file_path = os.path.join(self.gen_conf_ensemble_dir,
                                         file_name)
            merged_file_path = os.path.join(self.merged_conf_ensemble_dir,
                                            file_name)
            if not os.path.exists(gen_file_path) :
                print('Generating for ' + original_file_path)
                conf_ensemble = self.get_ensemble_from_file(original_file_path)
                rdkit_mol = conf_ensemble.mol
                ccdc_mol = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol)
                assert rdkit_mol.GetNumAtoms() == len(ccdc_mol.atoms)
                
                conformers = self.generate_conf_for_mol(ccdc_mol=ccdc_mol)
                gen_rdkit_mol, _ = self.ccdc_rdkit_connector.ccdc_ensemble_to_rdkit_mol(ccdc_ensemble=conformers, 
                                                                                        rdkit_mol=rdkit_mol,
                                                                                        generated=True,
                                                                                        remove_input_conformers=True)
                self.save_ensemble(rdkit_mol=gen_rdkit_mol,
                                   sd_writer_path=gen_file_path)
                
                merged_rdkit_mol = gen_rdkit_mol
                for conf in rdkit_mol.GetConformers() :
                    merged_rdkit_mol.AddConformer(conf, 
                                                  assignId=True)
                self.save_ensemble(rdkit_mol=merged_rdkit_mol,
                                   sd_writer_path=merged_file_path)
                
                
        except Exception as e :
            print('Generation failed for ' + original_file_path)
            print(str(e))
        
        
    def generate_conf_for_mol(self, 
                              ccdc_mol: Molecule,
                              n_confs: int = 250) -> ConformerHitList:
        """
        Generate conformers for input molecule
        Args:
            ccdc_mol: Molecule = CCDC molecule to generate confs for
            n_confs: int = maximum number of confs to generate
        Returns:
            ConformerHitList = confs for molecule
        """
        conf_generator = ConformerGenerator()
        conf_generator.settings.max_conformers = n_confs
        conformers = conf_generator.generate(ccdc_mol)
        return conformers

    
class ConfEnsembleLibraryTest(unittest.TestCase):
    
    def setUp(self):
        self.smiles = 'CC(=O)NC1=CC=C(C=C1)O'
        mols = [Chem.MolFromSmiles(self.smiles) for _ in range(2)]
        mols = [Chem.AddHs(mol, addCoords=True) for mol in mols]
        for mol in mols :
            EmbedMolecule(mol)
        mols = [Chem.RemoveHs(mol) for mol in mols]
        self.conf_ensemble = ConfEnsemble(mol_list=mols)
    
    def test_conf_ensemble_library(self):
        mol_list = []
        smiles_list = ['CC(=O)Nc1ccc(O)cc1', 
                       'CC(=O)Nc1ccc(O)cc1', 
                       'c1ccccc1']
        for smiles in smiles_list :
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol, addCoords=True)
            EmbedMultipleConfs(mol, 10)
            mol = Chem.RemoveHs(mol)
            mol_list.append(mol)
            
        conf_ensemble_library = ConfEnsembleLibrary.from_mol_list(mol_list)
        self.assertEqual(conf_ensemble_library.get_num_molecules(), 2)
        self.assertEqual(conf_ensemble_library.get_conf_ensemble('CC(=O)Nc1ccc(O)cc1').get_num_confs(), 20)
        self.assertEqual(conf_ensemble_library.get_conf_ensemble('c1ccccc1').get_num_confs(), 10)
        
if __name__ == '__main__':
    unittest.main()