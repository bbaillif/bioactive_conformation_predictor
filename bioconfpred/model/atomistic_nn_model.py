import torch
import torch.nn.functional as F
import os

from rdkit import Chem # safe import before any ccdc import
from rdkit.Chem import Mol

from torch_scatter import scatter
from torch_geometric.data import Batch, Data
# Differentiate PyGDataLoader and torch DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Any, Dict, List
from abc import abstractclassmethod
from bioconfpred.data.split import DataSplit
from bioconfpred.data.dataset import PyGDataset
from bioconfpred.data.featurizer import PyGFeaturizer
from .conf_pred_model import ConfPredModel
from .atomistic import AtomisticNN
from bioconfpred.params import LOG_DIRPATH
# from misc.mol_drawer import MolDrawer


class AtomisticNNModel(ConfPredModel):
    """
    Uses an atomistic neural network (NN) as a model to process an input conformation 
    to obtain a predicted value. In current work, we try to predict the ARMSD to 
    bioactive conformation
    
    :param config: Parameters of the model. Current list is:
        "num_interactions": int = 6, number of interaction blocks
        "cutoff": float = 10, cutoff for neighbourhood convolution graphs
        "lr": float = 1e-5, learning rate
        'batch_size': int = 256, batch size
        'data_split': DataSplit, object to split the dataset stored in the model
    :type config: Dict[str, Any]
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 atomistic_nn: AtomisticNN):
        ConfPredModel.__init__(self, config)
        self.data_split = config['data_split']
        
        self.atomistic_nn = atomistic_nn
        self.leaky_relu = torch.nn.LeakyReLU()
    
    
    @property
    def name(self) -> str:
        """
        :return: Name of the model
        :rtype: str
        """
        return 'AtomisticNNModel'
    
    
    @property
    def featurizer(self) -> PyGFeaturizer:
        """
        Featurizer used to transform a input molecule conformations to 
        a data_list interpretable by the model
        
        :return: Featurizer
        :rtype: PyGFeaturizer
        """
        return PyGFeaturizer()
        

    def forward(self, 
                batch: Batch) -> torch.Tensor:
        """
        Forward pass
        
        :param batch: PyG batch of atoms
        :type batch: torch_geometric.data.Batch
        :return: Predicted values
        :rtype: torch.Tensor (n_confs)
        """
        atomic_contributions = self.get_atomic_contributions(batch)
        pred = scatter(atomic_contributions, batch.batch, 
                       dim=0, reduce=self.atomistic_nn.readout)
        pred = self.leaky_relu(pred)
        return pred


    def record_loss(self,
                    batch: Batch,
                    loss_name: str = 'train_loss'
                    ) -> torch.Tensor:
        """
        Perform forward pass and records then returns the loss.
        Loss is MSELoss
        
        :param batch: PyG batch of atoms
        :type batch: torch_geometric.data.Batch
        :param loss_name: Name of the computed loss. Depends on whether the model
        is training, validating or testing
        :type loss_name: str
        :return: Loss
        :rtype: torch.tensor (1)
        """
        pred = self.forward(batch)
        y = batch.rmsd
        # y = batch.armsd
        
        # ARMSD can be -1 in case of error in splitting
        if not torch.all(y >= 0):
            import pdb;pdb.set_trace()
        loss = F.mse_loss(pred.squeeze(), y)
        self.log(loss_name, loss, batch_size=self.batch_size)
        return loss


    def get_preds_for_data_list(self,
                                data_list: List[Data]) -> None:
        """
        Get predictions for data_list (output from featurizer)
        
        :param data_list: List of PyG Data
        :type data_list: List[Data]
        :return: Predictions for each conformation
        :rtype: torch.Tensor (n_confs)
        """
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


    def set_dataset(self,
                    dataset: PyGDataset) -> None:
        """
        Set the dataset that the model with work with
        
        :param dataset: PyGDataset of featurized conformations
        :type dataset: PyGDataset
        """
        self.dataset = dataset
        self.subsets = self.dataset.get_split_subsets(self.data_split)


    def get_dataloader(self,
                       subset_name: str = 'train',
                       shuffle: bool = True) -> PyGDataLoader:
        """
        Obtain the dataloader corresponding to the subset of interest 
        (train, val or test)
        
        :param subset_name: 'train', 'val' or 'test'
        :type subset_name: str
        :param shuffle: Whether to shuffle the samples in DataLoader
        :type shuffle: bool
        :return: PyGDataLoader for the dataset
        :rtype: PyGDataLoader
        """
        assert subset_name in ['train', 'val', 'test']
        # self.dataset.add_bioactive_rmsds(self.data_split,
        #                                  subset_name)
        mol_ids, subset = self.subsets[subset_name]
        data_loader = PyGDataLoader(subset, 
                                    batch_size=self.batch_size, 
                                    shuffle=shuffle,
                                    num_workers=4)
        return data_loader
    
    
    def on_train_epoch_start(self) -> None:
        """
        Used to define the target for the dataset. The target depends of the subset
        For instance, a conformation can have a ARMSD of 0.5 in the training set,
        but a ARMSD of 1.5 in the test set because the train and test sets contains
        the same molecule but bound to different proteins.
        """
        self.dataset.add_bioactive_rmsds(data_split=self.data_split, 
                                         subset_name='train')
        self.mode = 'train'
    
    
    def on_validation_epoch_start(self) -> None:
        """
        Used to define the target for the dataset. The target depends of the subset
        For instance, a conformation can have a ARMSD of 0.5 in the training set,
        but a ARMSD of 1.5 in the test set because the val and test sets contains
        the same molecule but bound to different proteins.
        """
        self.dataset.add_bioactive_rmsds(data_split=self.data_split, 
                                         subset_name='val')
        self.mode = 'val'
        
    
    def on_test_epoch_start(self) -> None:
        """
        Used to define the target for the dataset. The target depends of the subset
        For instance, a conformation can have a ARMSD of 0.5 in the training set,
        but a ARMSD of 1.5 in the test set because the train and test sets contains
        the same molecule but bound to different proteins.
        """
        self.dataset.add_bioactive_rmsds(data_split=self.data_split, 
                                         subset_name='test')
        self.mode = 'test'
        
        
    def get_atomic_contributions(self, 
                                 batch: Batch) -> torch.Tensor:
        """
        Performs atomistic NN forward to obtain atomic contributions
        
        :param batch: Batch of featurized conformations
        :type batch: torch_geometric.data.Batch
        :return: Atomic contributions
        :rtype: torch.Tensor (n_atoms)
        """
        return self.atomistic_nn(batch.x.squeeze().long(), 
                                  batch.pos, 
                                  batch.batch)
         
         
    # def show_atomic_contributions(self, 
    #                               mol: Mol, 
    #                               save_dir: str=None) :
    #     """
    #     Draw atomic contributions for a molecule in a directory:
    #     one picture for each conformation, with coloring of atoms
    #     based on contribution
        
    #     :param mol: Molecule with conformations
    #     :type mol: Mol
    #     :param save_dir: Name of directory where to store images
    #     :type save_dir: str
    #     """
    #     data_list = self.featurizer.featurize_mol(mol)
    #     batch = Batch.from_data_list(data_list)
    #     batch.to(self.device)
    #     atomic_contributions = self.get_atomic_contributions(batch)
    #     atomic_contributions = atomic_contributions
    #     for batch_i in batch.batch.unique() :
    #         pred_is = atomic_contributions[batch.batch == batch_i]
    #         pred_is = pred_is.cpu().numpy().reshape(-1)
            
    #         MolDrawer().plot_values_for_mol(mol=mol,
    #                                         values=pred_is,
    #                                         suffix=batch_i,
    #                                         save_dir=save_dir)


    @staticmethod
    def get_model_log_name(data_split: DataSplit,
                           model_name: str):
        split_type = data_split.split_type
        split_i = data_split.split_i
        log_name = f'{split_type}_split_{split_i}_{model_name}'
        return log_name


    @classmethod
    def _get_model_for_data_split(cls,
                                 data_split: DataSplit,
                                 model_name: str,
                                 config: Dict[str, Any],
                                 log_dir: str = LOG_DIRPATH) -> 'AtomisticNNModel':
        """Get trained model for data split

        :param data_split: Data split
        :type data_split: DataSplit
        :param config: Configuration (parameter and value) for the model
        :type config: Dict[str, Any]
        :param log_dir: Directory where training log are stored
        :type log_dir: str, optional
        :return: Trained model
        :rtype: AtomisticNNModel
        """
        
        log_name = cls.get_model_log_name(data_split,
                                          model_name)
        checkpoint_dirname = os.path.join(log_dir,
                                          log_name,
                                          'checkpoints')
        checkpoint_filename = os.listdir(checkpoint_dirname)[0]
        checkpoint_path = os.path.join(checkpoint_dirname,
                                                  checkpoint_filename)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                                                    config=config)
        return model

    
    @abstractclassmethod
    def get_model_for_data_split(cls,
                                 data_split: DataSplit,
                                 log_dir: str = LOG_DIRPATH
                                 ) -> 'AtomisticNNModel':
        """Get the trained model for a given data split

        :param data_split: Data split
        :type data_split: DataSplit
        :param log_dir: Directory where training log are stored
        :type log_dir: str, optional
        :return: Trained model
        :rtype: AtomisticNNModel
        """
        pass
        

    