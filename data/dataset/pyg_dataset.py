import torch
import os
import random
import pandas as pd

from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from tqdm import tqdm
from typing import List
from .conf_ensemble_dataset import ConfEnsembleDataset
from data.utils import (PDBbind, 
                     LigandExpo)
from data.split import DataSplit
from data.preprocessing import (RMSDCalculator, ConfGenerator)
from data.featurizer import PyGFeaturizer
from conf_ensemble import ConfEnsembleLibrary
from params import (DATA_DIRPATH,
                    BIO_CONF_DIRNAME,
                    GEN_CONF_DIRNAME,
                    RMSD_DIRNAME,
                    PDBBIND_DIRPATH)


Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
random.seed(42)

class PyGDataset(ConfEnsembleDataset, InMemoryDataset) :
    """
    Create a torch geometric dataset for each conformations in the default
    conf ensemble library (bio+gen)
    
    :param cel_name: name of the bioactive conformations ensemble library, 
        defaults to 'pdb_conf_ensembles'
    :type cel_name: str, optional
    :param gen_cel_name: name of the generated conformers ensemble library, 
        defaults to 'gen_conf_ensembles'
    :type gen_cel_name: str, optional
    :param rmsd_name: name of the directory storing rmsd between bioactive
        conformations and generated conformers, defaults to 'rmsds'
    :type rmsd_name: str, optional
    :param root: Data directory,
    :type root: str, optional
    """
    
    def __init__(self,
                 cel_name: str = BIO_CONF_DIRNAME,
                 gen_cel_name: str = GEN_CONF_DIRNAME,
                 rmsd_name: str = RMSD_DIRNAME,
                 root: str = DATA_DIRPATH,
                 ) -> None:
        
        ConfEnsembleDataset.__init__(self, 
                                     cel_name=cel_name,
                                     gen_cel_name=gen_cel_name,
                                     root=root)
        self.rmsd_name = rmsd_name
        self.rmsd_dir = os.path.join(root, rmsd_name)
        
        # Creates the bioactive and generated conf ensemble libraries
        # And compute RMSD to bioactive
        if not os.path.exists(self.rmsd_dir):
            self.preprocess()
        
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
    
    
    def preprocess(self) -> None:
        """Preprocess data to prepare for the process function
        """
        
        pdbbind = PDBbind(root=PDBBIND_DIRPATH,
                            remove_unknown_ligand_name=True,
                            remove_unknown_uniprot=True)
        pdbbind_table = pdbbind.get_master_dataframe()
        
        ligand_expo = LigandExpo()

        pdbbind_ligands = pdbbind.get_ligands()
        print(len(pdbbind_ligands))

        d = {}
        for ligand in pdbbind_ligands :
            pdb_id = ligand.GetConformer().GetProp('PDB_ID')
            ligand_name = pdbbind_table[pdbbind_table['PDB code'] == pdb_id]['ligand name'].values[0]
            # Start the ensemble with the standard ligand that will act as template
            if not ligand_name in d :
                standard_ligand = ligand_expo.get_standard_ligand(ligand_name)
                d[ligand_name] = [standard_ligand]
            d[ligand_name].append(ligand)
                
        d.pop('YNU') # YNU ligands freezes the CCDC conformer generator 
                
        cel = ConfEnsembleLibrary.from_mol_dict(mol_dict=d)
        cel.save()

        cg = ConfGenerator()
        cg.generate_conf_for_library()
        
        rc = RMSDCalculator()
        rc.compute_rmsd_matrices()
    
    
    def process(self) -> None:
        """
        Creates the dataset from the bioactive+generated conf ensemble libraries
        """
        
        # Start generating data
        all_data_list = []
        self.mol_ids = []
        
        self.cel_df = pd.read_csv(self.cel_df_path)
        
        for i, row in tqdm(self.cel_df.iterrows(), total=self.cel_df.shape[0]) :
            name = row['ensemble_name']
            filename = row['filename']
            try :
                # Merge bio + conf ensembles
                ce = ConfEnsembleLibrary.get_merged_ce(filename, 
                                                       name,
                                                       cel_name1=self.cel_name,
                                                       cel_name2=self.gen_cel_name) 
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
            
            
    def add_bioactive_rmsds(self,
                            data_split: DataSplit,
                            subset_name: str) -> None:
        """Adds the bioactive rmsds to the dataset, based on precomputed rmsd
        files in the rmsd_name directory. Depending on the proteins in the 
        current set, the RMSD might be different, hence the choice
        of inputing the data split in that function

        :param data_split: Data split used in the modeling. 
        :type data_split: DataSplit
        :param subset_name: Name of the subset. train, val or test
        :type subset_name: str
        """
        rmsd_df = data_split.get_bioactive_rmsds(mol_id_df=self.mol_id_df,
                                                 subset_name=subset_name)
        df = self.mol_id_df.merge(rmsd_df.reset_index(), 
                                  on='mol_id',
                                  how='left') # in case RMSD is not defined for splits of a subset of the whole dataset (e.g. kinases)
        df = df.fillna(-1)
        rmsds = df['rmsd'].values
        self.data.rmsd = torch.tensor(rmsds, dtype=torch.float32)
        self.slices['rmsd'] = torch.arange(len(rmsds) + 1)
            
    
if __name__ == '__main__':
    pyg_dataset = PyGDataset()
    