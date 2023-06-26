import torch

from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import DimeNetPlusPlus
from .atomistic_nn import AtomisticNN


class AtomicDimeNet(AtomisticNN, DimeNetPlusPlus) :
    """Modification of the PyG DimeNetPlusPlus implementation to recover the atomic
    contributions to the prediction

        :param readout: Readout function to perform on the list of individual
        atomic values
        :type readout: str, optional
        :param hidden_channels: Size of hidden layers, defaults to 128
        :type hidden_channels: int, optional
        :param out_channels: Size of output layer, defaults to 1
        :type out_channels: int, optional
        :param num_blocks: Number of interaction blocks, defaults to 6
        :type num_blocks: int, optional
        :param int_emb_size: Interaction embedding size, defaults to 64
        :type int_emb_size: int, optional
        :param basis_emb_size: Size of basis function embedding, defaults to 8
        :type basis_emb_size: int, optional
        :param out_emb_channels: Size of output embedding, defaults to 256
        :type out_emb_channels: int, optional
        :param num_spherical: Number of spherical basis functions, defaults to 7
        :type num_spherical: int, optional
        :param num_radial: Number of radial basis functions, defaults to 6
        :type num_radial: int, optional
        """
    
    def __init__(self,
                 readout: str = 'add',
                 hidden_channels: int = 128,
                 out_channels: int = 1,
                 num_blocks: int = 6, 
                 int_emb_size: int = 64, 
                 basis_emb_size: int = 8,
                 out_emb_channels: int = 256, 
                 num_spherical: int = 7, 
                 num_radial: int = 6) :
        
        AtomisticNN.__init__(self,
                             readout=readout)
        DimeNetPlusPlus.__init__(self,
                                 hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                num_blocks=num_blocks, 
                                int_emb_size=int_emb_size, 
                                basis_emb_size=basis_emb_size,
                                out_emb_channels=out_emb_channels, 
                                num_spherical=num_spherical, 
                                num_radial=num_radial)
    
    def forward(self, 
                z: torch.Tensor, 
                pos: torch.Tensor, 
                batch=None):
        """
        Compute values for each atom in the input. Truncated version of
        forward function from DimeNetPlusPlus PyG implementation
        
        :param z: Sequence of atomic numbers 
        :type z: torch.Tensor (n_atoms)
        :param pos: Sequence of atomic cartesian positions
        :type pos: torch.Tensor (n_atoms, 3)
        :param batch: Sequence of batch identifier: which molecule the atom 
            corresponds to e.g. [0,0,0,1,1,1,1] means that the 3 first atoms
            belong to molecule 0, then the 4 next atoms belongs to molecule 1.
            If batch is None (default value), all atoms will be considered coming
            from the same molecule
        :type batch: torch.Tensor (n_atoms)
        """
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

# class BioDimeNet(pl.LightningModule):
#     def __init__(self, 
#                  batch_size: int=64, #given for the log function which is not extracting the batch size properly in my version of torch geometric
#                  task='rmsd'):
#         self.batch_size = batch_size
#         super().__init__()
#         self.dimenet = AtomicDimeNet()
#         self.leaky_relu = torch.nn.LeakyReLU()
#         self.sigmoid = torch.nn.Sigmoid()
        
#         assert task in ['rmsd', 'tfd']
#         self.task = task
        
#         #self.automatic_optimization=False

#     def forward(self, batch):
#         atomic_contributions = self.get_atomic_contributions(batch)
#         pred = scatter(atomic_contributions, batch.batch, 
#                        dim=0, reduce=self.dimenet.readout)
#         if self.task == 'rmsd' :
#             pred = self.leaky_relu(pred)
#         elif self.task == 'tfd' :
#             pred = self.sigmoid(pred)
#         return pred

#     def training_step(self, batch, batch_idx):
#         pred = self.forward(batch)
#         target = self._get_target(batch)
#         loss = F.mse_loss(pred.squeeze(), target)
#         self.log("train_loss", loss, batch_size=self.batch_size)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         pred = self.forward(batch)
#         target = self._get_target(batch)
#         loss = F.mse_loss(pred.squeeze(), target)
#         self.log("val_loss", loss, batch_size=self.batch_size)
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         pred = self.forward(batch)
#         target = self._get_target(batch)
#         loss = F.mse_loss(pred.squeeze(), target)
#         self.log("test_loss", loss, batch_size=self.batch_size)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_loss",
#                 "frequency": 1
#                 # If "monitor" references validation metrics, then "frequency" should be set to a
#                 # multiple of "trainer.check_val_every_n_epoch".
#             },
#         }
    
#     def _get_target(self, batch) :
#         if self.task == 'rmsd' :
#             return batch.rmsd
#         elif self.task == 'tfd' :
#             return batch.tfd
        
#     def get_atomic_contributions(self, batch):
#         return self.dimenet(batch.x.squeeze().long(), 
#                            batch.pos, 
#                            batch.batch)
         
#     def show_atomic_contributions(self, 
#                                   mol, 
#                                   save_dir: str=None) :
#         mol_featurizer = MoleculeFeaturizer()
#         data_list = mol_featurizer.featurize_mol(mol)
#         batch = Batch.from_data_list(data_list)
#         atomic_contributions = self.get_atomic_contributions(batch)
#         atomic_contributions = atomic_contributions
#         for batch_i in batch.batch.unique() :
#             pred_is = atomic_contributions[batch.batch == batch_i]
#             pred_is = pred_is.numpy().reshape(-1)
            
#             MolDrawer().plot_values_for_mol(mol=mol,
#                                             values=pred_is,
#                                             suffix=batch_i,
#                                             save_dir=save_dir)
            