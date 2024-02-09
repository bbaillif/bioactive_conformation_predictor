import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from rdkit.Chem.rdchem import Mol
from typing import Any, Dict
from torch import nn

from bioconfpred.model import ConfEnsembleModel 
from bioconfpred.data import E3FPDataset
from torch.utils.data import DataLoader
from bioconfpred.data.featurizer import E3FPFeaturizer
    
# TODO: write documentation (similar code architecture to bioschnet)
# but featurizer give different data_list type
    
class E3FPModel(ConfEnsembleModel):
    
    def __init__(self,
                 config: Dict[str, Any]):
        super().__init__(config)
        self.n_bits = config['n_bits']
        self.data_split = config['data_split']
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.n_bits, 10000),
            nn.ReLU(),
            nn.Linear(10000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.LeakyReLU(),
        )
        
        
    @property
    def name(self):
        return 'e3fp_model'
    
    
    @property
    def featurizer(self):
        return E3FPFeaturizer()
    
        
    def forward(self, x):
        pred = self.linear_layers(x)
        return pred

    def record_loss(self,
                    batch,
                    loss_name: str = 'train_loss'):
        x, y = batch
        pred = self.forward(x)
        loss = F.mse_loss(pred.squeeze(), y)
        self.log(loss_name, loss, batch_size=self.batch_size)
        return loss
    
    
    def set_dataset(self,
                    dataset: E3FPDataset) -> None:
        self.dataset = dataset
        self.subsets = self.dataset.get_split_subsets(self.data_split)
    
    
    def get_dataloader(self,
                       subset_name: str = 'train',
                       shuffle: bool = True):
        assert subset_name in ['train', 'val', 'test']
        mol_ids, subset = self.subsets[subset_name]
        data_loader = DataLoader(subset, 
                                 batch_size=self.batch_size, 
                                 shuffle=shuffle,
                                 num_workers=4)
        return data_loader

        
    def get_preds_for_data_list(self,
                                data_list):
        data_loader = DataLoader(data_list, 
                                    batch_size=self.batch_size)
        preds = None
        for x in data_loader:
            x = x.to(self.device)
            pred = self(x)
            pred = pred.detach() # .cpu().numpy().squeeze(1)
            if preds is None:
                preds = pred
            else:
                try:
                    preds = torch.cat((preds, pred))
                except: 
                    import pdb; pdb.set_trace()
        return preds
    