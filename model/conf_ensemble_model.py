import pytorch_lightning as pl
import torch

from abc import abstractmethod
from rdkit.Chem.rdchem import Mol
from typing import Type, Dict, Any
from data.conf_ensemble_dataset import ConfEnsembleDataset
from torch_geometric.data import DataLoader

from featurizer.mol_featurizer import MolFeaturizer

class ConfEnsembleModel(pl.LightningModule):
    
    def __init__(self, 
                 config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.lr = self.config['lr']
        self.batch_size = config['batch_size']
        
    
    @property
    @abstractmethod
    def featurizer(self) -> MolFeaturizer:
        pass
    
    
    @abstractmethod
    def forward(self) -> torch.Tensor:
        pass
    
    
    def get_preds_for_mol(self,
                          mol: Mol):
        data_list = self.featurizer.featurize_mol(mol)
        preds = self.get_preds_for_data_list(data_list)
        return preds
    
    
    @abstractmethod
    def get_preds_for_data_list(self,
                                data_list):
        pass
    
    @abstractmethod
    def get_dataloader(self,
                       split_name: str = 'train',
                       shuffle: bool = True):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


    def train_dataloader(self) -> DataLoader:
        data_loader = self.get_dataloader('train')
        return data_loader
    
    
    def val_dataloader(self) -> DataLoader:
        data_loader = self.get_dataloader('val', shuffle=False)
        return data_loader


    def test_dataloader(self) -> DataLoader:
        data_loader = self.get_dataloader('test', shuffle=False)
        return data_loader
    
    
    @abstractmethod
    def record_loss(self,
                    batch,
                    loss_name: str):
        pass
    
    
    def training_step(self, batch, batch_idx):
        loss = self.record_loss(batch, loss_name='train_loss')
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.record_loss(batch, loss_name='val_loss')
        return loss
    
    
    def test_step(self, batch, batch_idx):
        loss = self.record_loss(batch, loss_name='test_loss')
        return loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "val_loss",
            #     "frequency": 1
            # },
        }