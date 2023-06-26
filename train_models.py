import os
import pytorch_lightning as pl

from rdkit import Chem # safe import before ccdc imports
from data.dataset.pyg_dataset import PyGDataset
from data.split import RandomSplit, ScaffoldSplit
from model import SchNetModel, DimeNetModel, ComENetModel
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from params import (LOG_DIRPATH, 
                    SCHNET_CONFIG, 
                    DIMENET_CONFIG, 
                    COMENET_CONFIG)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

pl.seed_everything(42, workers=True)
log_dir = LOG_DIRPATH

# splits = ['random', 'scaffold', 'protein']
split_types = ['random', 'scaffold']
split_is = range(5)

dataset = PyGDataset()

model_configs = {SchNetModel: SCHNET_CONFIG,
                 DimeNetModel: DIMENET_CONFIG,
                 ComENetModel: COMENET_CONFIG}

for split_type, in split_types :
    
    for split_i in split_is :
    
        if split_type == 'random':
            data_split = RandomSplit(split_i=split_i)
        elif split_type == 'scaffold' :
            data_split = ScaffoldSplit(split_i=split_i)
            
        for model_class, config in model_configs.items():
        
            config['data_split'] = data_split
        
            model = model_class(config)
        
            if not os.path.exists(log_dir) :
                os.mkdir(log_dir)
            experiment_name = f'{split_type}_split_{split_i}_{model.name}'
            
            model.set_dataset(dataset)
            logger = TensorBoardLogger(save_dir=os.getcwd(), 
                                        version=experiment_name, 
                                        name=log_dir)
            early_stopping_callback = EarlyStopping(monitor="val_loss",
                                                    patience=5)
            checkpoint_callback = ModelCheckpoint(monitor='val_loss')
            trainer = pl.Trainer(logger=logger, 
                                max_epochs=50, 
                                gpus=1,
                                precision=16,
                                callbacks=[early_stopping_callback, checkpoint_callback])
            trainer.fit(model)
            trainer.test(model)