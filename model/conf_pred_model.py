import pytorch_lightning as pl
import torch

from abc import abstractmethod, ABC
from rdkit.Chem.rdchem import Mol
from typing import Dict, Any
from torch_geometric.data import DataLoader

from data.featurizer import MolFeaturizer

class ConfPredModel(pl.LightningModule, ABC):
    """
    Base class for models that take as input a molecular conformation
    and output a value
    
    :param config: dictionnary containing parameters for the model. Specific to
        the model, see heriting classes
    :type config: Dict[str, Any]
    """
    
    def __init__(self, 
                 config: Dict[str, Any]) -> None:
        pl.LightningModule.__init__(self)
        self.config = config
        self.lr = self.config['lr']
        self.batch_size = config['batch_size']
        
    
    @property
    @abstractmethod
    def featurizer(self) -> MolFeaturizer:
        """
        MolFeaturizer object used to convert a RDKit mol to appropriate 
        model input in a data_list format
        """
        pass
    
    
    @property
    @abstractmethod
    def name(self):
        """
        Name of the model
        """
        pass
    
    
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Forward pass of the model
        """
        pass
    
    
    @abstractmethod
    def get_preds_for_data_list(self,
                                data_list):
        """
        Get predictions for a data_list, output from a featurizer
        """
        pass
    
    
    @abstractmethod
    def get_dataloader(self,
                       split_name: str = 'train',
                       shuffle: bool = True):
        """
        Get dataloader for the stored dataset (allow Pytorch Lightning
        Trainer to fit and test, the model will work out which Data to use)
        """
        pass
    
    
    @abstractmethod
    def record_loss(self,
                    batch,
                    loss_name: str) -> torch.Tensor:
        """
        Record the loss in Tensorboard, and returns the loss
        """
        pass
    
    
    def get_preds_for_mol(self,
                          mol: Mol) -> torch.Tensor:
        """
        Get predicted values based on a RDKit molecule
        :param mol: RDKit mol containing conformations to get predictions for
        :type mol: Mol
        :return: Predictions for each conformation in the molecule
        :rtype: torch.Tensor
        """
        data_list = self.featurizer.featurize_mol(mol)
        preds = self.get_preds_for_data_list(data_list)
        return preds


    def train_dataloader(self) -> DataLoader:
        """
        Returns the train dataloader, uses the get_dataloader function
        specific to the model
        
        :rtype: DataLoader
        """
        data_loader = self.get_dataloader('train')
        return data_loader
    
    
    def val_dataloader(self) -> DataLoader:
        """
        Returns the val dataloader, uses the get_dataloader function
        specific to the model
        
        :rtype: DataLoader
        """
        data_loader = self.get_dataloader('val', shuffle=False)
        return data_loader


    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader, uses the get_dataloader function
        specific to the model
        
        :rtype: DataLoader
        """
        data_loader = self.get_dataloader('test', shuffle=False)
        return data_loader
    
    
    def training_step(self, 
                      batch: torch.Tensor, 
                      batch_idx: int):
        """
        Perform a training step: forward and record loss
        
        :param batch: batch of conformations in tensor format
        :type batch: torch.Tensor
        :param batch_idx: Index of the current batch (ith batch of the loader)
        :type batch_idx: int
        """
        loss = self.record_loss(batch, loss_name='train_loss')
        return loss


    def validation_step(self, 
                        batch: torch.Tensor, 
                        batch_idx: int):
        """
        Perform a validation step: forward and record loss
        
        :param batch: batch of conformations in tensor format
        :type batch: torch.Tensor
        :param batch_idx: Index of the current batch (ith batch of the loader)
        :type batch_idx: int
        """
        loss = self.record_loss(batch, loss_name='val_loss')
        return loss
    
    
    def test_step(self, 
                  batch: torch.Tensor, 
                  batch_idx: int):
        """
        Perform a test step: forward and record loss
        
        :param batch: batch of conformations in tensor format
        :type batch: torch.Tensor
        :param batch_idx: Index of the current batch (ith batch of the loader)
        :type batch_idx: int
        """
        loss = self.record_loss(batch, loss_name='test_loss')
        return loss
    
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer. Latest version is Adam, but a LR scheduler
        can be setup. See Pytorch Lightning
        
        :return: The optimizers and schedulers
        :rtype: Dict[str, Any]
        """
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