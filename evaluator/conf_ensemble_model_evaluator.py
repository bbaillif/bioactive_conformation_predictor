import os
import pickle
import numpy as np
import torch.nn.functional as F
import torch

from tqdm import tqdm
from rdkit.Chem.rdchem import Mol
from typing import List, Any, Dict, Sequence
from conf_ensemble.conf_ensemble_library import ConfEnsembleLibrary
from model import ConfEnsembleModel
from .evaluator import Evaluator
from torchmetrics import R2Score
from scipy.stats import pearsonr


class ConfEnsembleModelEvaluator(Evaluator):
    
    def __init__(self, 
                 model: ConfEnsembleModel,
                 evaluation_name: str,
                 results_dir: str = '/home/bb596/hdd/pdbbind_bioactive/results/'
                 ) -> None:
        Evaluator.__init__(self, 
                           evaluation_name, 
                           results_dir)
        self.model = model
        self.model.eval()
            
        self.model_dir = os.path.join(self.evaluation_dir, self.model.name)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.model_mol_results_path = os.path.join(self.model_dir, 
                                                   'model_mol_results.p')
        self.model_conf_results_path = os.path.join(self.model_dir, 
                                                    'model_conf_results.p')
            
        self.r2score = R2Score()
            
    def evaluate_library(self,
                         cel: ConfEnsembleLibrary,
                         d_targets: Dict[str, List[Any]]):
        
        assert cel.library.keys() == d_targets.keys()
            
        model_mol_results = {}
        model_conf_results = {}
            
        for name, ce in tqdm(cel.library.items()):
            try:
                targets = d_targets[name]
                assert ce.mol.GetNumConformers() == len(targets)
                results = self.evaluate_mol(ce.mol, targets)
                mol_results, conf_results = results
                model_mol_results[name] = mol_results
                model_conf_results[name] = conf_results
            except Exception as e:
                print(f'Evaluation failed for {name}')
                print(str(e))
                
        with open(self.model_mol_results_path, 'wb') as f:
            pickle.dump(model_mol_results, f)
        with open(self.model_conf_results_path, 'wb') as f:
            pickle.dump(model_conf_results, f)
            
            
    def evaluate_mol(self, 
                     mol: Mol,
                     targets: List[Any]):
        data_list = self.model.featurizer.featurize_mol(mol)
        mol_results, conf_results = self.evaluate_data_list(data_list, targets)
        return mol_results, conf_results
    
    
    def evaluate_data_list(self, 
                           data_list: List[Any],
                           targets: List[Any]):
        
        preds = self.model.get_preds_for_data_list(data_list)
        preds = preds.cpu().squeeze()
        
        mol_results, conf_results = self.evaluate_preds(preds, targets)
        
        return mol_results, conf_results
    
    def evaluate_preds(self,
                       preds: Sequence[Any],
                       targets: Sequence[Any]):
        
        try:
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
                targets = targets.to(preds) # preds can be on GPU initially
                
            preds = preds.view_as(targets)
            
            mol_results = {}
            conf_results = {}
            
            if preds.shape[0] > 1:
                mol_results['pearson'] = pearsonr(targets, preds)
                mol_results['r2'] = self.r2score(targets, preds)
            
            conf_results['targets'] = targets
            
            
            mae = F.l1_loss(targets, preds)
            mae = mae.cpu()
            mol_results['mae'] = mae.squeeze().item()
            
            mse = F.mse_loss(targets, preds)
            rmse = torch.sqrt(mse)
            rmse = rmse.cpu()
            mol_results['rmse'] = rmse.squeeze().item()
            
            preds = preds.cpu()
            conf_results['preds'] = preds.squeeze().tolist()
        
        except Exception as e:
            print(str(e))
            import pdb;pdb.set_trace()
        
        return mol_results, conf_results