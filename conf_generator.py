import os
import pandas as pd

from rdkit import Chem
from multiprocessing import Pool, TimeoutError
from ccdc_rdkit_connector import CcdcRdkitConnector
from ccdc.conformer import ConformerGenerator, ConformerHitList
from ccdc.molecule import Molecule
from ccdc.io import MoleculeWriter
from typing import Tuple

class ConfGenerator() :
    
    def __init__(self,
                 gen_cel_name: str = 'gen_conf_ensembles',
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/') -> None:
        self.gen_cel_name = gen_cel_name
        self.root = root
        self.gen_cel_dir = os.path.join(self.root, self.gen_cel_name)
        if not os.path.exists(self.gen_cel_dir) :
            os.mkdir(self.gen_cel_dir)
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
    
    def generate_conf_for_library(self,
                                  cel_name: str = 'pdb_conf_ensembles') :
        self.cel_name = cel_name
        self.cel_dir = os.path.join(self.root, self.cel_name)
        cel_df_path = os.path.join(self.cel_dir, 'ensemble_names.csv')
        cel_df = pd.read_csv(cel_df_path)
        params = list(zip(cel_df['ensemble_name'], 
                          cel_df['filename'], 
                          cel_df['smiles']))
            
        # for param in params:
        #     self.generate_conf_thread(param)
            
        with Pool(processes=12, maxtasksperchild=1) as pool :
            iterator = pool.imap(self.generate_conf_thread, params)
            done_looping = False
            while not done_looping:
                try:
                    results = iterator.next(timeout=120) 
                except StopIteration:
                    done_looping = True
                except TimeoutError:
                    print("Generation is too long, returning TimeoutError")
            
        gen_cel_df_path = os.path.join(self.gen_cel_dir, 'ensemble_names.csv')
        cel_df.to_csv(gen_cel_df_path)
            
            
    def generate_conf_thread(self, 
                             params: Tuple[str, str, str]) :
        """
        Thread for multiprocessing Pool to generate confs for a molecule using
        the CCDC conformer generator. Default behaviour is to save all
        generated confs in a dir, and initial+generated ensemble in a 'merged'
        dir
        Args:
            params: Tuple[str, str] = SMILES and SDF file name of the molecule
                to generate conformations for
        """
        
        name, filename, smiles = params
        print(smiles)
        gen_file_path = os.path.join(self.gen_cel_dir,
                                         filename)
        if not os.path.exists(gen_file_path):
            try :
                print('Generating for ' + name)
                ccdc_mol = Molecule.from_string(smiles)
                
                assert len(ccdc_mol.atoms) > 0
                
                conformers = self.generate_conf_for_mol(ccdc_mol=ccdc_mol)
                
                writer = MoleculeWriter(gen_file_path)
                for conformer in conformers:
                    writer.write_molecule(conformer.molecule)
                    
            except Exception as e :
                print(f'Generation failed for {name}')
                print(str(e))
        
        return None
        
        
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
    
    
if __name__ == '__main__' :
    cg = ConfGenerator()
    cg.generate_conf_for_library()