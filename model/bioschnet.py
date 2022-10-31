import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from rdkit import Chem # safe import before any ccdc import
from rdkit.Chem.rdchem import Mol
from torch_geometric.nn.models import SchNet
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from data import PyGDataset
from featurizer import PyGFeaturizer
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from mol_drawer import MolDrawer
from data import DataSplit
from typing import Any, Dict
from model.conf_ensemble_model import ConfEnsembleModel

class AtomicSchNet(SchNet) :
    
    def __init__(self, 
                 num_interactions: int = 6,
                 cutoff: float = 10):
        super().__init__(num_interactions=num_interactions, 
                         cutoff=cutoff)
    
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

class BioSchNet(ConfEnsembleModel):
    
    def __init__(self, 
                 config: Dict[str, Any]):
        super().__init__(config)
        self.data_split = config['data_split']
        
        num_interactions = config['num_interactions']
        cutoff = config['cutoff']
        self.schnet = AtomicSchNet(num_interactions=num_interactions,
                                   cutoff=cutoff)
        self.leaky_relu = torch.nn.LeakyReLU()
        
        
    @property
    def name(self):
        return 'bioschnet'
    
    
    @property
    def featurizer(self):
        return PyGFeaturizer()
        

    def forward(self, batch):
        atomic_contributions = self.get_atomic_contributions(batch)
        pred = scatter(atomic_contributions, batch.batch, 
                       dim=0, reduce=self.schnet.readout)
        pred = self.leaky_relu(pred)
        return pred


    def record_loss(self,
                    batch,
                    loss_name: str = 'train_loss'):
        pred = self.forward(batch)
        y = batch.rmsd
        loss = F.mse_loss(pred.squeeze(), y)
        self.log(loss_name, loss, batch_size=self.batch_size)
        return loss


    def set_dataset(self,
                    dataset: PyGDataset) -> None:
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
        
        
    def get_atomic_contributions(self, batch):
        return self.schnet(batch.x.squeeze().long(), 
                           batch.pos, 
                           batch.batch)
         
         
    def show_atomic_contributions(self, 
                                  mol, 
                                  save_dir: str=None) :
        data_list = self.featurizer.featurize_mol(mol)
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

        
    def get_preds_for_data_list(self,
                                data_list):
        with torch.inference_mode():
            data_loader = PyGDataLoader(data_list, 
                                        batch_size=self.batch_size)
            preds = None
            for batch in data_loader:
                batch.to(self.device)
                pred = self(batch)
                pred = pred.detach() # .cpu().numpy().squeeze(1)
                if preds is None:
                    preds = pred
                else:
                    preds = torch.cat((preds, pred))
            return preds
        

# https://docs.ray.io/en/releases-1.11.0/tune/tutorials/tune-pytorch-lightning.html#tuning-the-model-parameters
    
import math
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch_geometric.loader import DataLoader
from typing import List

# def train_bioschnet_tune(config, 
#                          train_loader: DataLoader,
#                          val_loader: DataLoader,
#                          num_epochs: int = 10, 
#                          num_gpus: float = 0):
#     model = BioSchNet(config)
#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         # If fractional GPUs passed in, convert to int.
#         gpus=math.ceil(num_gpus),
#         # progress_bar_refresh_rate=0,
#         # logger=TensorBoardLogger(
#         #     save_dir=tune.get_trial_dir(), name="", version="."),
#         # callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")]
#         )
#     trainer.fit(model, 
#                 train_dataloaders=train_loader, 
#                 val_dataloaders=val_loader)
    
def train_bioschnet_tune(config, 
                         num_epochs: int = 10, 
                         num_gpus: float = 0):
    model = BioSchNet(config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        progress_bar_refresh_rate=0,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")]
        )
    trainer.fit(model)
    

def tune_bioschnet_asha(data_split: DataSplit,
                        num_samples=20, 
                        num_epochs=10, 
                        gpus_per_trial=1):
    config = {
                "num_interactions": tune.choice(range(4, 8)),
                "cutoff": tune.choice(range(5, 12)),
                "lr": tune.loguniform(1e-5, 1e-3),
                'batch_size': 256,
                'data_split': data_split
            }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["num_interactions", "cutoff", "lr"],
        metric_columns=["loss", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_bioschnet_tune,
                                                    num_epochs=num_epochs,
                                                    num_gpus=gpus_per_trial)
    resources_per_trial = {"cpu": 6, "gpu": gpus_per_trial}

    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_bioschnet_asha",
        max_concurrent_trials=1)

    print("Best hyperparameters found were: ", analysis.best_config)