import pickle
import torch
import os
import random
import pandas as pd

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.separate import separate
from rdkit import Chem
from tqdm import tqdm
from typing import List
from molecule_encoders import MoleculeEncoders
from molecule_featurizer import MoleculeFeaturizer

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
random.seed(42)

class ConfEnsembleDataset(InMemoryDataset) :
    
    def __init__(self, 
                 root: str='data/',
                 dataset: str='pdbbind',
                 loaded_chunk: int=0,
                 smiles_list: List=[],
                 chunk_size: int=4000,
                 verbose: int=0):
        
        self.root = root
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.smiles_df = pd.read_csv(os.path.join(self.root, 'smiles_df.csv'))
        self.conf_df = pd.read_csv(os.path.join(self.root, 'conf_df.csv'))
        is_platinum = self.smiles_df['platinum']
        is_included = self.smiles_df['included']
        if dataset == 'pdbbind' :
            self.included_smiles = self.smiles_df[is_included & ~is_platinum]['smiles'].values
        else :
            self.included_smiles = self.smiles_df[is_included & is_platinum]['smiles'].values
        self.n_mols = len(self.included_smiles)
        self.n_chunks = int(self.n_mols / chunk_size) + 1
        self.encoder_path = os.path.join(self.root, 'molecule_encoders.p')
        
        with open(self.raw_paths[0], 'rb') as f :
            conf_ensemble_library = pickle.load(f)
        
        # Encoders
        if os.path.exists(self.encoder_path) : # Load existing encoder
            if self.verbose :
                print('Loading existing encoders')
            with open(self.encoder_path, 'rb') as f:
                self.molecule_encoders = pickle.load(f)
        else : # Create encoders
            if self.verbose :
                print('Creating molecule encoders')
            self.molecule_encoders = MoleculeEncoders()
            self.molecule_encoders.create_encoders(conf_ensemble_library)
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(self.molecule_encoders, f)
                
        self.molecule_featurizer = MoleculeFeaturizer(self.molecule_encoders)
        
        super().__init__(root=root) # calls the process functions
        
        self.load_chunk(loaded_chunk)
        
        # smiles filtering, useful for train/test split
        if len(smiles_list) :
            all_data_list = []
            for data in self :
                data_smiles = data.data_id
                if data_smiles in smiles_list :
                    all_data_list.append(data)
            self.data, self.slices = self.collate(all_data_list)
            
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['ccdc_generated_conf_ensemble_library.p']
    
    @property
    def processed_file_names(self) -> List[str]:
        if self.dataset == 'pdbbind' :
            return [f'pdbbind_dataset_{i}.pt' for i in range(self.n_chunks)]
        else :
            return [f'platinum_dataset_{i}.pt' for i in range(self.n_chunks)]
    
    def process(self):
        
        with open(self.raw_paths[0], 'rb') as f :
            conf_ensemble_library = pickle.load(f)
        
        # Generate data
        all_data_list = []
        for idx, smiles in enumerate(tqdm(self.included_smiles)) :
            conf_idxs = self.conf_df[self.conf_df['smiles'] == smiles].index
            
            conf_ensemble = conf_ensemble_library.get_conf_ensemble(smiles)
            mol = conf_ensemble.mol
            data_list = self.molecule_featurizer.featurize_mol(mol)
            rmsds = self.molecule_featurizer.get_bioactive_rmsds(mol)
            for i, data in enumerate(data_list) :
                data.rmsd = rmsds[i]
                all_data_list.append(data)
            
            
#             try :
#                 conf_ensemble = conf_ensemble_library.get_conf_ensemble(smiles)
#                 mol = conf_ensemble.mol
#                 data_list = self.molecule_featurizer.featurize_mol(mol)
#                 rmsds = self.molecule_featurizer.get_bioactive_rmsds(data_list)
#                 for i, data in enumerate(data_list) :
#                     data.rmsd = rmsds[i]
#                     all_data_list.append(data)
#             except Exception as e : 
#                 print('Error for the smiles : ' + smiles)
#                 print(type(e))

            if (idx + 1) % self.chunk_size == 0 :
                chunk_number = int(((idx + 1) / self.chunk_size) - 1)
                torch.save(self.collate(all_data_list), self.processed_paths[chunk_number])
                if self.verbose :
                    print(f'Chunk num {chunk_number} saved')
                all_data_list = []
        chunk_number = chunk_number + 1
        torch.save(self.collate(all_data_list), self.processed_paths[chunk_number])
        if self.verbose :
            print(f'Chunk num {chunk_number} saved')
        all_data_list = []
            
    def _angle_interpolation(self, start, end, amounts=[0.5]) :
        interps = []
        for amount in amounts :
            shortest_angle = ((((end - start) % 360) + 540) % 360) - 180
            to_add = shortest_angle * amount
            interps.append((((start + to_add) + 180) % 360) - 180)
        return interps
    
    def load_chunk(self, chunk_number) :
        self.data, self.slices = torch.load(self.processed_paths[chunk_number])
        if self.verbose :
            print(f'Chunk num {chunk_number} loaded')
        
    def get(self, idx: int) -> Data:
        
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        return data
    
if __name__ == '__main__':
    conf_ensemble_dataset = ConfEnsembleDataset()
    del conf_ensemble_dataset
    conf_ensemble_dataset = ConfEnsembleDataset(dataset='platinum')
    