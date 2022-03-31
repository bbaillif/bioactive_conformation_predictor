import os
import pickle
import copy
import unittest
import pandas as pd

from typing import List
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
from rdkit.Chem.rdmolfiles import SDWriter, SDMolSupplier
from ccdc_rdkit_connector import CcdcRdkitConnector
from ccdc.conformer import ConformerGenerator
from multiprocessing import Pool
from tqdm import tqdm
from conf_ensemble import ConfEnsemble

class ConfEnsembleLibrary(object) :
    
    def __init__(self, 
                 conf_ensemble_dir: str='data/conf_ensembles/',
                 conf_ensemble_df_name: str='index.csv',
                 gen_conf_ensemble_dir: str='data/gen_conf_ensembles/',
                 merged_conf_ensemble_dir: str='data/all_conf_ensembles/',
                 unique_file_ensemble_path: str='data/all_conf_ensembles.sdf') :
        self.conf_ensemble_dir = conf_ensemble_dir
        self.conf_ensemble_df_name = conf_ensemble_df_name
        self.gen_conf_ensemble_dir = gen_conf_ensemble_dir
        self.merged_conf_ensemble_dir = merged_conf_ensemble_dir
        self.unique_file_ensemble_path = unique_file_ensemble_path
        
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
                      standardize_mols: bool=True) :
        cel = cls()
        print('Generating library')
        for mol in tqdm(mol_list) :
            Chem.AssignAtomChiralTagsFromStructure(mol) # could be changed to AssignStereochemistryFrom3D
            smiles = Chem.MolToSmiles(mol)
            try :
                if smiles in cel.library :
                    cel.library[smiles].add_confs_from_mol(mol,
                                                           standardize_mol=standardize_mols)
                else :
                    cel.library[smiles] = ConfEnsemble(mol=mol,
                                                       standardize_mols=standardize_mols)
            except Exception as e :
                print('Molecule failed')
                print(str(e))
                
        return cel
    
    
    @classmethod
    def from_dirs(cls,
                  dir_pathes: list) :
        print('Generating library')
        mol_list = []
        for dir_path in dir_pathes:
            file_names = os.listdir(dir_path)
            for file_name in file_names :
                file_path = os.path.join(dir_path, file_name)
                mol_list.extend([mol for mol in SDMolSupplier(file_path)])
        return cls.from_mol_list(mol_list=mol_list)
              
    @classmethod  
    def from_pickle(cls,
                    file_path: str) :
        with open(file_path, 'rb') as f :
            cel = pickle.load(f) 
        return cel
                
                
    def load(self, load_dir=None) :
        if load_dir is None :
            load_dir = self.conf_ensemble_dir
        elif load_dir == 'merged' :
            load_dir = self.merged_conf_ensemble_dir
        elif load_dir == 'generated' :
            load_dir = self.gen_conf_ensemble_dir
        self.load_metadata(load_dir=load_dir)
            
        for smiles, file_name in tqdm(self.table.values):
            try :
                file_path = os.path.join(load_dir, file_name)
                conf_ensemble = self.load_ensemble_from_file(file_path=file_path)
                if smiles in self.library :
                    self.library[smiles].add_confs_from_mol(mol=conf_ensemble.mol,
                                                            standardize_mol=False)
                else :
                    self.library[smiles] = conf_ensemble
                
                # conf_ensemble = self.load_ensemble(file_path=file_path)
                # basename, file_name = os.path.split(file_path)
                # if subset == 'merged' :
                #     gen_file_path = os.path.join(self.gen_conf_ensemble_dir,
                #                                  file_name)
                #     if os.path.exists(gen_file_path) :
                #         gen_conf_ensemble = self.load_ensemble_from_file(file_path=gen_file_path)
                #         conf_ensemble.add_confs_from_mol(mol=gen_conf_ensemble.mol,
                #                                          standardize_mol=False)
                # self.library[smiles] = conf_ensemble
            except Exception as e :
                print('Error with ' + file_path)
                print(str(e))
               
               
    def load_metadata(self,
                      load_dir=None) :
        # if load_dir is None :
        #     load_dir = self.conf_ensemble_df_path
        # elif load_dir == 'merged' :
        #     load_dir = self.merged_conf_ensemble_dir
        self.table = pd.read_csv(self.conf_ensemble_df_path, index_col=0)
               
               
    def load_ensemble_from_smiles(self,
                                 smiles,
                                 load_dir=None) :
        if load_dir is None :
            load_dir = self.conf_ensemble_dir
        elif load_dir == 'generated' :
            load_dir = self.gen_conf_ensemble_dir
        elif load_dir == 'merged' :
            load_dir = self.merged_conf_ensemble_dir
            
        smiles_table = self.table[self.table['smiles'] == smiles]
        if len(smiles_table) :
            file_name = smiles_table['file_name'].values[0]
            file_path = os.path.join(load_dir,
                                     file_name)
            return self.load_ensemble_from_file(file_path=file_path)
        else :
            raise Exception('Smiles not in conf ensemble')
               
               
    def load_ensemble_from_file(self, file_path):
        try :
            suppl = SDMolSupplier(file_path)
            return ConfEnsemble(mol_list=suppl,
                                standardize_mols=False)
        except Exception as e:
            print(str(e))
            raise Exception(f'{file_path} is not ok')
               
                
    def save(self, 
             save_dir=None,
             save_table=True,
             unique_file=False) :
        if save_dir is None :
            save_dir = self.conf_ensemble_dir
        elif save_dir == 'merged' :
            save_dir = self.merged_conf_ensemble_dir
        
        if unique_file :
            writer = SDWriter(self.unique_file_ensemble_path)
        
        i = 0
        for smiles, ce in tqdm(self.library.items()):
            mol = ce.mol
            file_name = f'{i}.sdf'
            if unique_file :
                self.save_confs_to_writer(rdkit_mol=mol,
                                          writer=writer)
            else :
                writer_path = os.path.join(save_dir,
                                       file_name)
                self.save_ensemble(rdkit_mol=mol,
                                file_path=writer_path)
            row = pd.DataFrame([[smiles, file_name]], 
                               columns=self.table.columns)
            self.table = pd.concat([self.table, row], ignore_index=True)
            i = i + 1
            
        if unique_file :
            writer.close()

        if save_table :
            self.table.to_csv(self.conf_ensemble_df_path)


    def save_ensemble(self,
                      rdkit_mol,
                      file_path) :
        with SDWriter(file_path) as sd_writer :
            self.save_confs_to_writer(rdkit_mol=rdkit_mol,
                                      writer=sd_writer)


    def save_confs_to_writer(self,
                             rdkit_mol,
                             writer) :
        for conf in rdkit_mol.GetConformers() :
            conf_id = conf.GetId()
            for prop, value in conf.GetPropsAsDict().items():
                rdkit_mol.SetProp(prop, str(value))
            writer.write(rdkit_mol, conf_id)
            for prop in rdkit_mol.GetPropsAsDict() :
                rdkit_mol.ClearProp(prop)


    def get_num_molecules(self) :
        return len(self.library)
    
    
    def get_conf_ensemble(self, smiles, canonical_check=False) :
        if canonical_check :
            canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            assert canon_smiles in self.library, \
            'Canonical input smiles not in library'
        return self.library[smiles]
    
    
    def get_unique_molecules(self) :
        return self.library.items()
    
    
    def merge(self, second_library) :
        new_cel = copy.deepcopy(self)
        for smiles, conf_ensemble in second_library.get_unique_molecules() :
            mol = conf_ensemble.mol
            if smiles in self.library :
                new_cel.library[smiles].add_confs_from_mol(mol)
            else :
                new_cel.library[smiles] = ConfEnsemble(mol=mol)
        return new_cel
    
    
    def remove_smiles(self, smiles) :
        if smiles in self.library :
            self.library.pop(smiles)
        else :
            print(f'Input smiles {smiles} not in library')
        
        
    def generate_conf_pool(self, 
                           included_smiles) :
        params = []
    
        included_table = self.table[self.table['smiles'].isin(included_smiles)]
        for i, row in included_table.iterrows() :
            smiles = row['smiles']
            file_name = row['file_name']
            params.append((smiles, file_name))
            
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.generate_conf_thread, params)
            
    def generate_conf_thread(self, params) :
        
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
                conf_ensemble = self.load_ensemble_from_file(original_file_path)
                rdkit_mol = conf_ensemble.mol
                #import pdb;pdb.set_trace()
                ccdc_mol = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol)
                assert rdkit_mol.GetNumAtoms() == len(ccdc_mol.atoms)
                
                conformers = self.generate_conf_for_mol(ccdc_mol=ccdc_mol)
                gen_rdkit_mol, generated_conf_ids = self.ccdc_rdkit_connector.ccdc_ensemble_to_rdkit_mol(ccdc_ensemble=conformers, 
                                                                                                        rdkit_mol=rdkit_mol,
                                                                                                        generated=True,
                                                                                                        remove_input_conformers=True)
                self.save_ensemble(rdkit_mol=gen_rdkit_mol,
                                   file_path=gen_file_path)
                
                merged_rdkit_mol = gen_rdkit_mol
                for conf in rdkit_mol.GetConformers() :
                    merged_rdkit_mol.AddConformer(conf, 
                                                  assignId=True)
                self.save_ensemble(rdkit_mol=merged_rdkit_mol,
                                   file_path=merged_file_path)
                
                
        except Exception as e :
            print('Generation failed for ' + original_file_path)
            print(str(e))
        
        
    def generate_conf_for_mol(self, ccdc_mol) :
        conf_generator = ConformerGenerator()
        conf_generator.settings.max_conformers = 100
        conformers = conf_generator.generate(ccdc_mol)
        return conformers
        
    # @classmethod
    # def to_path(cls,
    #             cel,
    #             path: str='data/raw/ccdc_generated_conf_ensemble_library.p') :
    #     with open(path, 'wb') as f :
    #         pickle.dump(cel, f)
    
    # def save(self,
    #          path: str='data/raw/ccdc_generated_conf_ensemble_library.p') :
    #     with open(path, 'wb') as f :
    #         pickle.dump(self, f)
        
    # @classmethod
    # def from_path(cls, 
    #               path: str='data/raw/ccdc_generated_conf_ensemble_library.p') :
    #     with open(path, 'rb') as f:
    #         conf_ensemble_library = pickle.load(f)
    #     return conf_ensemble_library
    
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
            
        conf_ensemble_library = ConfEnsembleLibrary(mol_list)
        self.assertEqual(conf_ensemble_library.get_num_molecules(), 2)
        self.assertEqual(conf_ensemble_library.get_conf_ensemble('CC(=O)Nc1ccc(O)cc1').get_num_confs(), 20)
        self.assertEqual(conf_ensemble_library.get_conf_ensemble('c1ccccc1').get_num_confs(), 10)
        
if __name__ == '__main__':
    unittest.main()