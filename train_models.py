import os
import pytorch_lightning as pl
import pandas as pd

from rdkit import Chem # safe import before ccdc imports
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from conf_ensemble_dataset_in_memory import ConfEnsembleDataset
from data_split import DataSplit, MoleculeSplit, ProteinSplit
from bioschnet import BioSchNet
from molsize_model import MolSizeModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

pl.seed_everything(42, workers=True)

splits = ['protein']

def get_loaders(dataset: Dataset, 
                data_split: DataSplit,
                batch_size: int=64) :
    
    
    subsets = dataset.get_splits(data_split)

    train_loader = DataLoader(subsets['train'], batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(subsets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(subsets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

for split_type in splits :
    
    for split_i in range(5) :
    
        if split_type in ['random', 'scaffold'] :
            data_split = MoleculeSplit(split_type, split_i)
        elif split_type == 'protein' :
            data_split = ProteinSplit(split_type, split_i)
            
        batch_size = 128
        dataset = ConfEnsembleDataset()
        train_loader, val_loader, test_loader = get_loaders(dataset, 
                                                            data_split, 
                                                            batch_size)
        
        experiment_name = f'{split_type}_split_{split_i}'
        if not experiment_name in os.listdir('lightning_logs') :
            litschnet = BioSchNet()
            logger = TensorBoardLogger(save_dir=os.getcwd(), version=experiment_name, name="lightning_logs")
            early_stopping_callback = EarlyStopping(monitor="val_loss")
            trainer = pl.Trainer(logger=logger, 
                                 max_epochs=20, 
                                 gpus=1,
                                 callbacks=[early_stopping_callback])
            trainer.fit(litschnet, train_loader, val_loader)
            trainer.test(litschnet, test_loader)
        del train_loader, val_loader, test_loader