import pickle
import torch
import os
import random
import pandas as pd

from torch_geometric.data import Dataset, Data, Batch
from rdkit import Chem
from tqdm import tqdm
from typing import List
from molecule_encoders import MoleculeEncoders
from molecule_featurizer import MoleculeFeaturizer

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
random.seed(42)

class ConfEnsembleDataset(Dataset) :
    
    def __init__(self, 
                 root: str='data/'):
        
        self.root = root
        self.smiles_df = pd.read_csv(os.path.join(self.root, 'smiles_df.csv'))
        self.conf_df = pd.read_csv(os.path.join(self.root, 'conf_df.csv'))
        self.included_smiles = self.smiles_df[self.smiles_df['included']]['smiles'].values
        self.n_confs = len(self.conf_df[self.conf_df['smiles'].isin(self.included_smiles)])
        self.encoder_path = os.path.join(self.root, 'molecule_encoders.p')
        
        with open(self.raw_paths[0], 'rb') as f :
            conf_ensemble_library = pickle.load(f)
        
        # Encoders
        if os.path.exists(self.encoder_path) : # Load existing encoder
            print('Loading existing encoders')
            with open(self.encoder_path, 'rb') as f:
                self.molecule_encoders = pickle.load(f)
        else : # Create encoders
            print('Creating molecule encoders')
            self.molecule_encoders = MoleculeEncoders()
            self.molecule_encoders.create_encoders(conf_ensemble_library)
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(self.molecule_encoders, f)
                
        self.molecule_featurizer = MoleculeFeaturizer(self.molecule_encoders)
        
        super().__init__(root=root) # calls the process functions
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['ccdc_generated_conf_ensemble_library.p']
    
    @property
    def processed_file_names(self) -> List[str]:
        return [f'data_{idx}.pt' for idx in range(self.n_confs)]
    
    def process(self):
        
        with open(self.raw_paths[0], 'rb') as f :
            conf_ensemble_library = pickle.load(f)
        
        # Generate data
        for smiles in tqdm(self.included_smiles) :
            conf_idxs = self.conf_df[self.conf_df['smiles'] == smiles].index
            try :
                conf_ensemble = conf_ensemble_library.get_conf_ensemble(smiles)
                mol = conf_ensemble.mol
                data_list = self.molecule_featurizer.featurize_mol(mol)
                rmsds = self.molecule_featurizer.get_bioactive_rmsds(data_list)
                for i, data in enumerate(data_list) :
                    data.rmsd = rmsds[i]
                    torch.save(data, os.path.join(self.processed_dir, f'data_{conf_idxs[i]}.pt'))
            except Exception as e : 
                print('Error for the smiles : ' + smiles)
                print(type(e))
            
    def _angle_interpolation(self, start, end, amounts=[0.5]) :
        interps = []
        for amount in amounts :
            shortest_angle = ((((end - start) % 360) + 540) % 360) - 180
            to_add = shortest_angle * amount
            interps.append((((start + to_add) + 180) % 360) - 180)
        return interps
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
if __name__ == '__main__':
    conf_ensemble_dataset = ConfEnsembleDataset()