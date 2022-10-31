import os
import numpy as np
import pandas as pd

from rdkit.Chem.rdchem import Conformer
from .conf_ensemble_dataset import ConfEnsembleDataset
from .data_split import DataSplit
from typing import Tuple
from tqdm import tqdm

class RDKitMolDataset(ConfEnsembleDataset):
    
    def __init__(self, 
                 cel_name: str = 'pdb_conf_ensembles_moe_all',
                 gen_cel_name: str = 'gen_conf_ensembles_moe_all',
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/',
                 generated_only: bool = False) -> None:
        ConfEnsembleDataset.__init__(self, 
                                     cel_name=cel_name,
                                     gen_cel_name=gen_cel_name,
                                     root=root)
        self.generated_only = generated_only
        
        self.mol_id_df_filename = f'rdkit_mol_ids.csv'
        self.mol_id_df_path = os.path.join(self.root, self.mol_id_df_filename)
        
        # self.cel_moe_dir = os.path.join(self.root, 'pdb_conf_ensembles_moe/')
        # self.gen_cel_moe_dir = os.path.join(self.root, 'gen_conf_ensembles_moe_all/')
        
        if not os.path.exists(self.mol_id_df_path) :
            self.process()
        self.load()
    
    
    def process(self):
        all_mol_ids = []
        print('Computing mol ids')
        for i, row in tqdm(self.cel_df.iterrows(), total=self.cel_df.shape[0]):
            name = row['ensemble_name']
            filename = row['filename']
            ce = self.get_merged_ce(filename, name)
            mol_ids = self.compute_mol_ids(ce.mol, name)
            all_mol_ids.extend(mol_ids)
        
        self.mol_id_df = pd.DataFrame({'mol_id' : all_mol_ids})
        self.mol_id_df.to_csv(self.mol_id_df_path)
    
    
    def load(self):
        self.confs = []
        print('Loading confs')
        # cel_df = self.cel_df.iloc[:100]
        cel_df = self.cel_df
        names = []
        for i, row in tqdm(cel_df.iterrows(), total=cel_df.shape[0]):
            name = row['ensemble_name']
            names.append(name)
            filename = row['filename']
            try:
                if self.generated_only:
                    ce = self.get_generated_ce(filename, name)
                    confs = [conf for conf in ce.mol.GetConformers()]
                else:
                    ce = self.get_merged_ce(filename, name)
                    confs = [conf for conf in ce.mol.GetConformers()]
                self.confs.extend(confs)
            except:
                print(f'Loading failed for {name}')
        
        self.mol_id_df = pd.read_csv(self.mol_id_df_path, index_col=0)
        # self.mol_id_df['name'] = self.mol_id_df['mol_id'].apply(lambda s: s.split('_')[0])
        # self.mol_id_df = self.mol_id_df[self.mol_id_df['name'].isin(names)]
        if self.generated_only:
            self.mol_id_df = self.mol_id_df[self.mol_id_df['mol_id'].str.contains('Gen')]
            self.mol_id_df = self.mol_id_df.reset_index()
            
        try:
            assert self.mol_id_df.shape[0] == len(self.confs)
        except:
            import pdb;pdb.set_trace()
            
    
    def add_bioactive_rmsds(self,
                            data_split: DataSplit) -> None:
        rmsds = self.get_bioactive_rmsds(data_split)
        self.rmsds = np.array(rmsds)
        
        
    def __getitem__(self, 
                    index) -> Tuple[Conformer, float]:
        if isinstance(index, (tuple)):
            confs = []
            for i in index:
                confs.append(self.confs[i])
        else:
            confs = self.confs[index]
        if hasattr(self, 'rmsds'):
            rmsd = self.rmsds[index]
        else:
            rmsd = 0
        return confs, rmsd
    
    def __len__(self) -> int:
        return len(self.confs)
    
    
def rdkit_mol_dataset_collate(data_list):
    confs = []
    rmsds = []
    for data in data_list:
        confs.append(data[0])
        rmsds.append(data[1])
    return confs, rmsds
    
if __name__ == '__main__' :
    dataset = RDKitMolDataset()
    
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, 
                    batch_size=128,
                    collate_fn=rdkit_mol_dataset_collate)
    try:
        for b in dl:
            print(b)
    except:
        import pdb; pdb.set_trace()