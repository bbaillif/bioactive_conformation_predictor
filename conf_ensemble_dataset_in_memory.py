import torch
import os
import random
import pandas as pd

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.separate import separate
from rdkit import Chem
from tqdm import tqdm
from typing import List
from molecule_featurizer import MoleculeFeaturizer
from conf_ensemble_library import ConfEnsembleLibrary
from data_split import DataSplit

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
random.seed(42)

class ConfEnsembleDataset(InMemoryDataset) :
    
    def __init__(self, 
                 data_split: DataSplit,
                 dataset: str='train',
                 root: str='data/',
                 filter_out_bioactive: bool=False):
        
        self.data_split = data_split
        assert dataset in ['train', 'val', 'test']
        self.dataset = dataset
        self.root = root
        self.filter_out_bioactive = filter_out_bioactive
        
        self.smiles_df = pd.read_csv(os.path.join(self.root, 'smiles_df.csv'))
        
        is_included = self.smiles_df['included']
        self.included_data = self.smiles_df[is_included]
        self.included_smiles = self.included_data['smiles'].unique()
        
        self.molecule_featurizer = MoleculeFeaturizer()
        
        super().__init__(root=root) # calls the process functions
        
    @property
    def raw_file_names(self) -> List[str]:
        return ['smiles_df.csv']
    
    @property
    def processed_file_names(self) -> List[str]:
        split_type = self.data_split.split_type
        split_i = self.data_split.split_i
        return [f'pdbbind_{split_type}_{split_i}_{self.dataset}.pt']
    
    
    def process(self):
        
        cel = ConfEnsembleLibrary()
        cel.load_metadata()
        
        # There is a 1:1 relationship between PDB id and smiles
        # While there is a 1:N relationship between smiles and PDB id
        # The second situation is able to select only certain bioactive conformation
        # for a molecule, while the first situation takes all pdb_ids for a smiles
        if self.data_split.split_type in ['random', 'scaffold'] :
            processed_smiles = self.data_split.get_smiles(dataset=self.dataset)
            processed_pdb_ids = self.get_pdb_ids_for_smiles(processed_smiles)
        elif self.data_split.split_type in ['protein'] :
            processed_pdb_ids = self.data_split.get_pdb_ids(dataset=self.dataset)
            processed_smiles = self.get_smiles_for_pdb_ids(processed_pdb_ids)
        
        processed_smiles = [smiles 
                           for smiles in processed_smiles
                           if smiles in self.included_smiles]
        
        # Generate data
        all_data_list = []
        for smiles in tqdm(processed_smiles) :
            try :
                conf_ensemble = cel.load_ensemble_from_smiles(smiles,
                                                            load_dir='merged')
                mol = conf_ensemble.mol
            
                mol_ids = []
                gen_i = 0
                confs = [conf for conf in mol.GetConformers()]
                for conf in confs :
                    if conf.HasProp('Generator') :
                        mol_id = f'{smiles}_Gen_{gen_i}'
                        mol_ids.append(mol_id)
                        gen_i = gen_i + 1
                    else :
                        pdb_id = conf.GetProp('PDB_ID')
                        if pdb_id in processed_pdb_ids :
                            mol_id = f'{smiles}_{pdb_id}'
                            mol_ids.append(mol_id)
                        else : # can only happen in the protein split situation
                            mol.RemoveConformer(conf.GetId())
                    
                data_list = self.molecule_featurizer.featurize_mol(mol, 
                                                                mol_ids=mol_ids)
                rmsds = self.molecule_featurizer.get_bioactive_rmsds(mol)
                for i, data in enumerate(data_list) :
                    data.rmsd = rmsds[i]
                    all_data_list.append(data)
            except Exception as e :
                print(f'Error processing {smiles}')
                print(e)
                
        torch.save(self.collate(all_data_list), self.processed_paths[0])
            
            
    def get_smiles_for_pdb_ids(self,
                               pdb_ids: list=[]) :
        assert all([len(pdb_id) == 4 for pdb_id in pdb_ids])
        processed_data = self.smiles_df[self.smiles_df['id'].isin(pdb_ids)]
        processed_smiles = processed_data['smiles'].unique()
        return processed_smiles
            
            
    def get_pdb_ids_for_smiles(self,
                               smiles_list: list=[]) :
        processed_data = self.smiles_df[self.smiles_df['smiles'].isin(smiles_list)]
        processed_pdb_ids = processed_data['id'].unique()
        return processed_pdb_ids
    
    
if __name__ == '__main__':
    conf_ensemble_dataset = ConfEnsembleDataset()
    