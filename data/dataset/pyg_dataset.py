import torch
import os
import random
import pandas as pd

from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from tqdm import tqdm
from typing import List
from .conf_ensemble_dataset import ConfEnsembleDataset
from .pdbbind import PDBbindMetadataProcessor
from .pdb_ligand_expo import PDBLigandExpo
from .split.data_split import DataSplit
from .rmsd_calculator import RMSDCalculator
from featurizer import PyGFeaturizer
from data.conf_generator import ConfGenerator
from conf_ensemble import ConfEnsembleLibrary


Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
random.seed(42)

class PyGDataset(ConfEnsembleDataset, InMemoryDataset) :
    """
    Create a torch geometric dataset for each conformations in the default
    conf ensemble library (bio+gen)
    Args:
        :param root: Directory where library is stored
        :type root: str
    """
    
    def __init__(self,
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/'):
        
        ConfEnsembleDataset.__init__(self, 
                                     root=root)
        
        # pdbbind_df_path = os.path.join(self.root, 'pdbbind_df.csv')
        # self.pdbbind_df = pd.read_csv(pdbbind_df_path)
        
        self.molecule_featurizer = PyGFeaturizer()
        
        self.mol_id_df_filename = f'pyg_mol_ids.csv'
        self.mol_id_df_path = os.path.join(self.root, self.mol_id_df_filename)
        
        InMemoryDataset.__init__(self, root=root) # calls the process function via the InMemoryDataset class
        assert os.path.exists(self.mol_id_df_path)
        self.mol_id_df = pd.read_csv(self.mol_id_df_path, index_col=0)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        return [f'pdbbind_data.pt']
    
    
    def process(self):
        """
        Creates the dataset from the default library
        """
        
        # Creates the bioactive and generated conf ensemble libraries
        # self.preprocess()
        
        # Start generating data
        all_data_list = []
        self.mol_ids = []
        
        for i, row in tqdm(self.cel_df.iterrows(), total=self.cel_df.shape[0]) :
            name = row['ensemble_name']
            filename = row['filename']
            try :
                ce = self.get_merged_ce(filename, name) # Merge bio + conf ensembles
                mol = ce.mol
            
                mol_ids = self.compute_mol_ids(mol, name) # Give ids to recognize each conf
                    
                data_list = self.molecule_featurizer.featurize_mol(mol, 
                                                                   mol_ids=mol_ids)
                all_data_list.extend(data_list)
                self.mol_ids.extend(mol_ids)
                
            except Exception as e :
                print(f'Error processing {name}')
                print(str(e))
                
        self.mol_id_df = pd.DataFrame({'mol_id' : self.mol_ids})
        self.mol_id_df.to_csv(self.mol_id_df_path)
                
        torch.save(self.collate(all_data_list), self.processed_paths[0])
            
            
    def preprocess(self):
        
        pmp = PDBbindMetadataProcessor(root='/home/bb596/hdd/PDBBind/',
                                       remove_unknown_ligand_name=True,
                                       remove_unknown_uniprot=True)
        pdbbind_table = pmp.get_master_dataframe()
        
        ple = PDBLigandExpo()

        pdbbind_ligands = pmp.get_ligands()
        print(len(pdbbind_ligands))

        d = {}
        for ligand in pdbbind_ligands :
            pdb_id = ligand.GetConformer().GetProp('PDB_ID')
            ligand_name = pdbbind_table[pdbbind_table['PDB code'] == pdb_id]['ligand name'].values[0]
            # Start the ensemble with the standard ligand that will act as template
            if not ligand_name in d :
                standard_ligand = ple.get_standard_ligand(ligand_name)
                d[ligand_name] = [standard_ligand]
            d[ligand_name].append(ligand)
                
        d.pop('YNU') # YNU ligands freezes the CCDC conformer generator 
                
        cel = ConfEnsembleLibrary.from_mol_dict(mol_dict=d)
        cel.save()
        cel.create_pdbbind_df()

        cg = ConfGenerator()
        cg.generate_conf_for_library()
        
        rc = RMSDCalculator()
        rc.compute_rmsd_matrices()
            
            
    def add_bioactive_rmsds(self,
                            data_split: DataSplit,
                            subset_name: str) :
        data_split.set_dataset(self)
        rmsd_df = data_split.get_bioactive_rmsds(subset_name)
        df = self.mol_id_df.merge(rmsd_df.reset_index(), 
                                  on='mol_id',
                                  how='left') # in case RMSD is not defined for splits of a subset of the whole dataset (e.g. kinases)
        df = df.fillna(-1)
        rmsds = df['rmsd'].values
        self.data.rmsd = torch.tensor(rmsds, dtype=torch.float32)
        self.slices['rmsd'] = torch.arange(len(rmsds) + 1)
            
    
if __name__ == '__main__':
    pyg_dataset = PyGDataset()
    