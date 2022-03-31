import copy
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
from conf_ensemble_library import ConfEnsembleLibrary

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
random.seed(42)

class ConfEnsembleDataset(InMemoryDataset) :
    
    def __init__(self, 
                 root: str='data/',
                 dataset: str='pdbbind',
                 loaded_chunk: int=0,
                 smiles_list: List=[],
                 pdb_ids_list: List=[],
                 chunk_size: int=4000,
                 verbose: int=0,
                 filter_out_bioactive: bool=False):
        
        assert dataset in ['pdbbind', 'platinum']
        
        self.root = root
        self.dataset = dataset
        self.loaded_chunk = loaded_chunk
        self.smiles_list = smiles_list
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.filter_out_bioactive = filter_out_bioactive
        
        self.smiles_df = pd.read_csv(os.path.join(self.root, 'smiles_df.csv'))
        
        in_platinum = self.smiles_df['dataset'] == 'platinum'
        in_pdbbind = self.smiles_df['dataset'] == 'pdbbind'
        is_included = self.smiles_df['included']
        platinum_smiles = self.smiles_df[in_platinum]['smiles'].unique()
        platinum_pdb_ids = self.smiles_df[in_platinum]['pdb_id'].unique()
        smiles_in_platinum = self.smiles_df['smiles'].isin(platinum_smiles)
        pdb_id_in_platinum = self.smiles_df['pdb_id'].isin(platinum_pdb_ids)
        if self.dataset == 'pdbbind' :
            self.processed_data = self.smiles_df[is_included 
                                                  & in_pdbbind 
                                                  & ~smiles_in_platinum
                                                  & ~pdb_id_in_platinum]  
        else :
            self.processed_data = self.smiles_df[is_included & in_platinum]
        self.processed_smiles = self.processed_data['smiles'].unique()
        
        self.n_mols = len(self.processed_smiles)
        self.n_chunks = int(self.n_mols / chunk_size) + 1
        
        self.molecule_featurizer = MoleculeFeaturizer()
        
        super().__init__(root=root) # calls the process functions
        
        self.load_chunk(loaded_chunk)
        
        if self.dataset == 'pdbbind' :
            smiles_list = [smiles 
                           for smiles in smiles_list 
                           if smiles not in platinum_smiles]
            pdb_ids_list = [pdb_id 
                            for pdb_id in pdb_ids_list 
                            if pdb_id not in platinum_pdb_ids]
            
        # If we have pdb ids input only, we add smiles_list to have the Gen conformations data
        if len(pdb_ids_list) and not len(smiles_list) :
            id_in_list = self.smiles_df['id'].isin(pdb_ids_list)
            smiles_list = self.smiles_df[id_in_list]['smiles'].unique()
        
        # smiles or pdb_id filtering, useful for train/test split
        if len(smiles_list) or len(pdb_ids_list):
            self.filter_data(smiles_list, pdb_ids_list)
           
           
    @property
    def raw_file_names(self) -> List[str]:
        return ['all_conf_ensemble_library.p']
    
    @property
    def processed_file_names(self) -> List[str]:
        if self.dataset == 'pdbbind' :
            #return [f'pdbbind_dataset_{i}.pt' for i in range(self.n_chunks)]
            return [f'pdbbind_dataset_{i}.pt' for i in range(4)]
        else :
            return [f'platinum_dataset_{i}.pt' for i in range(self.n_chunks)]
    
    
    def process(self):
        
        cel = ConfEnsembleLibrary()
        cel.load_metadata()
        #cel.load('merged')
        
        # Generate data
        all_data_list = []
        chunk_number = 0
        for idx, smiles in enumerate(tqdm(self.processed_smiles)) :
            conf_ensemble = cel.load_ensemble_from_smiles(smiles,
                                                          load_dir='merged')
            mol = conf_ensemble.mol
            if self.dataset == 'pdbbind' :
                try :
                    mol_ids = []
                    gen_i = 0
                    for conf in mol.GetConformers() :
                        pdb_id = conf.GetProp('PDB_ID')
                        if conf.HasProp('Generator') :
                            mol_id = f'{smiles}_Gen_{gen_i}'
                            gen_i = gen_i + 1
                        else :
                            mol_id = f'{smiles}_{pdb_id}'
                        mol_ids.append(mol_id)
                    data_list = self.molecule_featurizer.featurize_mol(mol, 
                                                                    mol_ids=mol_ids)
                    rmsds = self.molecule_featurizer.get_bioactive_rmsds(mol)
                    for i, data in enumerate(data_list) :
                        data.rmsd = rmsds[i]
                        all_data_list.append(data)

                    if (idx + 1) % self.chunk_size == 0 :
                        torch.save(self.collate(all_data_list), self.processed_paths[chunk_number])
                        if self.verbose :
                            print(f'Chunk num {chunk_number} saved')
                        all_data_list = []
                        chunk_number = chunk_number + 1
                except Exception as e :
                    print(f'Error with smiles {smiles}')
                    print(str(e))
            elif self.dataset == 'platinum' : 
                try :
                    new_mol = copy.deepcopy(mol)
                    new_mol.RemoveAllConformers()
                    mol_ids = []
                    gen_i = 0
                    for conf in mol.GetConformers() :
                        pdb_id = conf.GetProp('PDB_ID')
                        if conf.HasProp('Generator') :
                            new_mol.AddConformer(conf, assignId=True)
                            mol_id = f'{smiles}_Gen_{gen_i}'
                            mol_ids.append(mol_id)
                            gen_i = gen_i + 1
                        elif conf.HasProp('platinum_id') :
                            new_mol.AddConformer(conf, assignId=True)
                            mol_id = f'{smiles}_{pdb_id}'
                            mol_ids.append(mol_id)
                    data_list = self.molecule_featurizer.featurize_mol(new_mol, 
                                                                    mol_ids=mol_ids)
                    rmsds = self.molecule_featurizer.get_bioactive_rmsds(mol)
                    for i, data in enumerate(data_list) :
                        data.rmsd = rmsds[i]
                        all_data_list.append(data)

                    if (idx + 1) % self.chunk_size == 0 :
                        torch.save(self.collate(all_data_list), self.processed_paths[chunk_number])
                        if self.verbose :
                            print(f'Chunk num {chunk_number} saved')
                        all_data_list = []
                        chunk_number = chunk_number + 1
                except Exception as e :
                    print(f'Error with smiles {smiles}')
                    print(str(e))
                
        torch.save(self.collate(all_data_list), self.processed_paths[chunk_number])
        
        if self.verbose :
            print(f'Chunk num {chunk_number} saved')
            
        all_data_list = []
            
            
    def filter_data(self, 
                    smiles_list,
                    pdb_ids_list) :
        # if pdb_ids_list, smiles_list is corresponding to the one seen for pdbs
        all_data_list = []
        for data in self :
            data_id_split = data.data_id.split('_')
            data_smiles = data_id_split[0]
            data_pdb_id = data_id_split[1].replace('e+', 'e').replace('30.0', '3e01')
            is_bioactive = data_pdb_id != 'Gen'
            is_generated = not is_bioactive
            smiles_in_list = data_smiles in smiles_list
            
            is_included = True
            if len(pdb_ids_list) :
                if not data_pdb_id in pdb_ids_list :
                    if not (smiles_in_list and is_generated) :
                        is_included = False
            elif len(smiles_list) :
                if not smiles_in_list :
                    is_included = False

            if is_included :
                all_data_list.append(data)
                
        self.data, self.slices = self.collate(all_data_list)
            
            
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
    