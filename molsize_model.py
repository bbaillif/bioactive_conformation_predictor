import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch import nn

class MolSizeModel(pl.LightningModule) :
    
    def __init__(self):
        super().__init__()
        self.linear_layers = nn.Sequential(nn.Linear(2, 100),
                                           nn.LeakyReLU(),
                                           nn.Linear(100, 100),
                                           nn.LeakyReLU(),
                                           nn.Linear(100, 1),
                                           nn.LeakyReLU())
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, batch):
        n_heavy_atoms = batch.n_heavy_atoms
        n_rotatable_bonds = batch.n_rotatable_bonds
        mol_size_descriptors = torch.cat([n_heavy_atoms.reshape((-1, 1)), 
                                          n_rotatable_bonds.reshape((-1, 1))],
                                         dim=1).float()
        #import pdb;pdb.set_trace()
        pred = self.linear_layers(mol_size_descriptors)
        return pred

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = self._get_target(batch)
        loss = F.mse_loss(pred.squeeze(), target)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = self._get_target(batch)
        loss = F.mse_loss(pred.squeeze(), target)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = self._get_target(batch)
        loss = F.mse_loss(pred.squeeze(), target)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
    
    def _get_target(self, batch) :
        return batch.rmsd