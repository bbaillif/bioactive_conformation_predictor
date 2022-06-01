import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import copy
import os

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.nn.models import SchNet
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from molecule_featurizer import MoleculeFeaturizer
from torch_geometric.data import Batch
from mol_drawer import MolDrawer

class AtomicSchNet(SchNet) :
    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        return h

class LitSchNet(pl.LightningModule):
    def __init__(self, task='rmsd'):
        super().__init__()
        self.schnet = AtomicSchNet()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        assert task in ['rmsd', 'tfd']
        self.task = task
        
        #self.automatic_optimization=False

    def forward(self, batch):
        atomic_contributions = self.get_atomic_contributions(batch)
        pred = scatter(atomic_contributions, batch.batch, 
                       dim=0, reduce=self.schnet.readout)
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
        
    def get_atomic_contributions(self, batch):
        return self.schnet(batch.x.squeeze().long(), 
                           batch.pos, 
                           batch.batch)
         
    def show_atomic_contributions(self, 
                                  mol, 
                                  save_dir: str=None) :
        mol_featurizer = MoleculeFeaturizer()
        data_list = mol_featurizer.featurize_mol(mol)
        batch = Batch.from_data_list(data_list)
        atomic_contributions = self.get_atomic_contributions(batch)
        atomic_contributions = atomic_contributions
        for batch_i in batch.batch.unique() :
            pred_is = atomic_contributions[batch.batch == batch_i]
            pred_is = pred_is.numpy().reshape(-1)
            
            MolDrawer().plot_values_for_mol(mol=mol,
                                            values=pred_is,
                                            suffix=batch_i,
                                            save_dir=save_dir)
            