import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch_geometric.nn.models import SchNet

class LitSchNet(pl.LightningModule):
    def __init__(self, task='rmsd'):
        super().__init__()
        self.schnet = SchNet()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        assert task in ['rmsd', 'tfd']
        self.task = task
        
        #self.automatic_optimization=False

    def forward(self, batch):
        pred = self.schnet(batch.z.long(), batch.pos, batch.batch)
        if self.task == 'rmsd' :
            pred = self.leaky_relu(pred)
        elif self.task == 'tfd' :
            pred = self.sigmoid(pred)
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
        if self.task == 'rmsd' :
            return batch.rmsd
        elif self.task == 'tfd' :
            return batch.tfd