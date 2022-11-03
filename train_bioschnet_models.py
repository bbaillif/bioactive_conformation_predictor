import os
import pytorch_lightning as pl

from rdkit import Chem # safe import before ccdc imports
from data.pyg_dataset import PyGDataset
from data.split.data_split import MoleculeSplit, ProteinSplit
from model.bioschnet import BioSchNet
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

pl.seed_everything(42, workers=True)

splits = ['random', 'scaffold', 'protein']

for split_type in splits :
    
    for split_i in range(5) :
    
        if split_type in ['random', 'scaffold'] :
            data_split = MoleculeSplit(split_type, split_i)
        elif split_type == 'protein' :
            data_split = ProteinSplit(split_type, split_i)
            
        config = {"num_interactions": 6,
                  "cutoff": 10,
                  "lr":1e-4,
                  'batch_size': 256,
                  'data_split': data_split}
        dataset = PyGDataset()
        log_dir = 'lightning_logs'
        if not os.path.exists(log_dir) :
            os.mkdir(log_dir)
        experiment_name = f'{split_type}_split_{split_i}'
        # if not experiment_name in os.listdir(log_dir) :
        os.system(f'rm -r {os.path.join(log_dir, experiment_name)}')
        bioschnet = BioSchNet(config)
        bioschnet.set_dataset(dataset)
        logger = TensorBoardLogger(save_dir=os.getcwd(), 
                                    version=experiment_name, 
                                    name=log_dir)
        early_stopping_callback = EarlyStopping(monitor="val_loss",
                                                patience=5)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss')
        trainer = pl.Trainer(logger=logger, 
                            max_epochs=20, 
                            gpus=1,
                            precision=16,
                            callbacks=[early_stopping_callback, checkpoint_callback])
        trainer.fit(bioschnet)
        trainer.test(bioschnet)