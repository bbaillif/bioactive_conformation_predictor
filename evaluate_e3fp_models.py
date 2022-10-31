import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from conf_ensemble import ConfEnsembleLibrary, ConfEnsemble
from data.e3fp_dataset import E3FPDataset
# from evaluator import ConfEnsembleModelEvaluator
from evaluator.ranker_evaluator import RankerEvaluator
from model import E3FPModel
from data import MoleculeSplit, ProteinSplit
from rankers import ModelRanker

def get_model_from_checkpoint(experiment_name, data_split) :
    checkpoint_dir = os.path.join('lightning_logs', experiment_name, 'checkpoints')
    checkpoint_name = os.listdir(checkpoint_dir)[0]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    config = {'n_bits': 4096, 
                  'batch_size': 256,
                  'lr': 1e-6,
                  'data_split': data_split}
    model = E3FPModel.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
    return model


root = '/home/bb596/hdd/pdbbind_bioactive/data'
cel_df = pd.read_csv(os.path.join(root, 
                                  'pdb_conf_ensembles', 
                                  'ensemble_names.csv'))

dataset = E3FPDataset()
with torch.inference_mode():
    for split_type in ['protein', 'random', 'scaffold']:
        
        for split_i in range(5):
            
            experiment_name = f'{split_type}_split_{split_i}'
            
            if split_type == 'protein':
                data_split = ProteinSplit(split_i=split_i)
            else:
                data_split = MoleculeSplit(split_type, split_i)
            
            subsets = dataset.get_split_subsets(data_split)
            mol_ids, subset = subsets['test']
            
            model = get_model_from_checkpoint(f'{split_type}_split_{split_i}_e3fp',
                                            data_split)
            ranker = ModelRanker(model, use_cuda=True)

            evaluator = RankerEvaluator(ranker, evaluation_name=experiment_name)
            evaluator.evaluate_subset(subset, mol_ids)