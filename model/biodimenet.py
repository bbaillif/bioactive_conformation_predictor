import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from molecule_featurizer import MoleculeFeaturizer
from torch_geometric.data import Batch
from mol_drawer import MolDrawer

from torch_geometric.nn.models import DimeNetPlusPlus


class AtomicDimeNet(DimeNetPlusPlus) :
    
    def __init__(self) :
        super().__init__(hidden_channels=128,
                         out_channels=1,
                         num_blocks=6, 
                         int_emb_size=64, 
                         basis_emb_size=8,
                         out_emb_channels=256, 
                         num_spherical=7, 
                         num_radial=6)
        self.readout = 'add'
    
    def forward(self, z, pos, batch=None):
        """"""
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return P

class BioDimeNet(pl.LightningModule):
    def __init__(self, 
                 batch_size: int=64, #given for the log function which is not extracting the batch size properly in my version of torch geometric
                 task='rmsd'):
        self.batch_size = batch_size
        super().__init__()
        self.dimenet = AtomicDimeNet()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        assert task in ['rmsd', 'tfd']
        self.task = task
        
        #self.automatic_optimization=False

    def forward(self, batch):
        atomic_contributions = self.get_atomic_contributions(batch)
        pred = scatter(atomic_contributions, batch.batch, 
                       dim=0, reduce=self.dimenet.readout)
        if self.task == 'rmsd' :
            pred = self.leaky_relu(pred)
        elif self.task == 'tfd' :
            pred = self.sigmoid(pred)
        return pred

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = self._get_target(batch)
        loss = F.mse_loss(pred.squeeze(), target)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = self._get_target(batch)
        loss = F.mse_loss(pred.squeeze(), target)
        self.log("val_loss", loss, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = self._get_target(batch)
        loss = F.mse_loss(pred.squeeze(), target)
        self.log("test_loss", loss, batch_size=self.batch_size)
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
        return self.dimenet(batch.x.squeeze().long(), 
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
            