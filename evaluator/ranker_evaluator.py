from collections import defaultdict
import numpy as np
import os
import pickle
from data.e3fp_dataset import E3FPDataset

from evaluator import ConfEnsembleModelEvaluator
from tqdm import tqdm
from .evaluator import Evaluator
from torch.utils.data import Subset
from rankers import ModelRanker
from model import SchNetModel
from data.split import MoleculeSplit
from rankers import (ConfRanker, ModelRanker, PropertyRanker)
from typing import Sequence, List, Dict, Any
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.ML.Scoring.Scoring import CalcEnrichment, CalcBEDROC
from conf_ensemble import ConfEnsembleLibrary
from scipy.stats import spearmanr

class RankerEvaluator(Evaluator):
    
    def __init__(self, 
                 ranker: ConfRanker,
                 evaluation_name: str,
                 results_dir: str = '/home/bb596/hdd/pdbbind_bioactive/results/',
                 bioactive_threshold: float = 1,
                 non_bioactive_threshold: float = 2.5) :
        Evaluator.__init__(self, evaluation_name, results_dir)
        self.ranker = ranker
        self.bioactive_threshold = bioactive_threshold
        self.non_bioactive_threshold = non_bioactive_threshold
        
        self.ranker_dir = os.path.join(self.evaluation_dir, self.ranker.name)
        if not os.path.exists(self.ranker_dir):
            os.mkdir(self.ranker_dir)
        self.ranker_mol_results_path = os.path.join(self.ranker_dir, 
                                                    'ranker_mol_results.p')
        self.ranker_conf_results_path = os.path.join(self.ranker_dir, 
                                                     'ranker_conf_results.p')
        
        self.ef_step = 0.01
        self.ef_fractions = np.arange(0 + self.ef_step,
                                      1 + self.ef_step,
                                      self.ef_step)
        self.ef_fractions = np.around(self.ef_fractions, 2)
        
        if isinstance(self.ranker, ModelRanker) :
            self.model_evaluator = ConfEnsembleModelEvaluator(self.ranker.model,
                                                              evaluation_name,
                                                              results_dir)


    def evaluate_library(self,
                         cel: ConfEnsembleLibrary,
                         d_targets: Dict[str, List[Any]]):
        assert cel.library.keys() == d_targets.keys()
            
        ranker_mol_results = {}
        ranker_conf_results = {}
            
        for name, ce in tqdm(cel.library.items()):
            try:
                targets = d_targets[name]
                assert ce.mol.GetNumConformers() == len(targets)
                results = self.evaluate_mol(ce.mol, targets)
                mol_results, conf_results = results
                ranker_mol_results[name] = mol_results
                ranker_conf_results[name] = conf_results
            except Exception as e:
                print(f'Evaluation failed for {name}')
                print(str(e))
                
        with open(self.ranker_mol_results_path, 'wb') as f:
            pickle.dump(ranker_mol_results, f)
        with open(self.ranker_conf_results_path, 'wb') as f:
            pickle.dump(ranker_conf_results, f)

    
    def evaluate_mol(self,
                     mol: Mol,
                     targets: Sequence[float]):
        
        input_list = self.ranker.get_input_list_from_mol(mol)   
        all_mol_results, all_conf_results = self.evaluate_input_list(input_list, targets)
        return all_mol_results, all_conf_results
               
               
    def evaluate_input_list(self,
                            input_list,
                            targets):
        
        try:
        
            all_mol_results = {}
            all_conf_results = {}
            
            # Including bioactive confs
            values = self.ranker.get_values(input_list)
            ranks = self.ranker.get_ranks(values)
            rmsds = targets
            rmsds = np.array(rmsds)
            
            all_conf_results['all_ranks'] = ranks.tolist()
            
            if isinstance(self.ranker, ModelRanker) :
                results = self.model_evaluator.evaluate_preds(preds=values, 
                                                            targets=targets)
                mol_results, conf_results = results
                all_mol_results['regression'] = mol_results
                all_conf_results['regression'] = conf_results
            
            is_bioactive = rmsds == 0
            
            if np.sum(is_bioactive) > 0 and self.ranker.name != 'CCDC':
                bioactive_results = self.get_bioactive_ranking(values,
                                                            ranks,
                                                            is_bioactive)
                all_mol_results['bioactive'] = bioactive_results
            
            # Generated confs only
            generated_mask = rmsds > 0
            # input_list = [data 
            #             for i, data in enumerate(input_list)
            #             if i in np.argwhere(generated_mask)]
            # values = self.ranker.get_values(input_list)
            values = values[generated_mask]
            rmsds = rmsds[generated_mask]
            ranks = self.ranker.get_ranks(values)
            
            if values.shape[0] > 1:
                all_mol_results['spearman'] = spearmanr(rmsds, values)
            
            all_conf_results['generated_ranks'] = ranks.tolist()
            
            bioactive_like_idxs = np.argwhere(rmsds <= self.bioactive_threshold).squeeze(1)
            non_bioactive_idxs = np.argwhere(rmsds >= self.non_bioactive_threshold).squeeze(1)
            d_mask_idxs = {'bioactive_like' : bioactive_like_idxs,
                            'non_bioactive' : non_bioactive_idxs}
            
            all_conf_results['bioactive_like_idxs'] = bioactive_like_idxs.tolist()
            all_conf_results['non_bioactive'] = non_bioactive_idxs.tolist()
            
            for label, mask_idxs in d_mask_idxs.items():
                n_labelled = mask_idxs.shape[0]
                ratio = n_labelled / rmsds.shape[0]
                if ratio > 0 and ratio < 1: # ratio = 1 is useless to measure
                    mask = [True 
                            if i in mask_idxs 
                            else False 
                            for i in range(len(rmsds))]
                    label_results = self.get_mask_ranking(values,
                                                        ranks, 
                                                        mask)
                    all_mol_results[label] = label_results
                    
        except Exception as e:
            print('Error in input list evaluation')
            print(str(e))
        
        return all_mol_results, all_conf_results
                
    def get_mask_ranking(self,
                         values: Sequence[float],
                         ranks: Sequence[int],
                         mask: Sequence[bool],
                         enrichment: bool = True):
        results = {}
        assert len(ranks) == len(mask)
        
        try:
            n_confs = len(ranks)
            n_masked = np.sum(mask)
                
            results['n_masked'] = n_masked
            results['n_confs'] = n_confs
                
            mask_ranks = ranks[mask]
            results['first_rank'] = float(mask_ranks.min())
            
            normalized_ranks = mask_ranks / n_confs
            results['normalized_first_rank'] = float(normalized_ranks.min())
            
            if enrichment:
                results['ef'] = {}
                results['normalized_ef'] = {}
                results['old_normalized_ef'] = {}
                results['recall'] = {}
                
                # transform into RDKit ranking metrics
                values = np.array(values)
                mask = np.array(mask)
                sorting = values.argsort()
                sorted_values = values[sorting]
                sorted_mask = mask[sorting]
                ranked_list = np.array(list(zip(sorted_values, sorted_mask)))
                
                old_max_ef = n_confs / n_masked
                for fraction in self.ef_fractions :
                    n_confs_fraction = int(n_confs * fraction)
                    n_confs_fraction = max(n_confs_fraction, 1) # avoid division by zero
                    best_ratio = min(n_masked, n_confs_fraction) / n_confs_fraction
                    max_ef = best_ratio / (n_masked / n_confs)
                    ef_result = CalcEnrichment(ranked_list, 
                                                col=1, 
                                                fractions=[fraction])
                    if not len(ef_result) :
                        if fraction < 0.5:
                            import pdb;pdb.set_trace()
                        ef = 1
                    else :
                        ef = ef_result[0]
                    normalized_ef = ef / max_ef
                    
                    results['ef'][fraction] = ef
                    results['normalized_ef'][fraction] = normalized_ef
                    
                    old_normalized_df = ef / old_max_ef
                    results['old_normalized_ef'][fraction] = old_normalized_df
                    
                    i = int(np.ceil(n_confs * fraction))
                    fraction_ranks = ranks[:i]
                    ratio_mask = np.isin(fraction_ranks, mask_ranks).sum()
                    recall = ratio_mask / n_masked
                    results['recall'][fraction] = recall
                    
                bedroc = CalcBEDROC(ranked_list, col=1, alpha=20)
                results['bedroc'] = bedroc
                
        except Exception as e:
            print('Error in mask ranking')
            print(str(e))
            # import pdb; pdb.set_trace()
        
        return results
                                 
    def get_bioactive_ranking(self,
                              values,
                              ranks,
                              is_bioactive):
        results = self.get_mask_ranking(values, 
                                        ranks, 
                                        mask=is_bioactive,
                                        enrichment=False)
        return results


    def evaluate_subset(self,
                        subset: Subset,
                        mol_ids: List[str]):
            
        ranker_mol_results = {}
        ranker_conf_results = {}
        
        grouped_data, d_targets = self.group_by_name(subset, mol_ids)
        assert grouped_data.keys() == d_targets.keys()
        if isinstance(subset.dataset, E3FPDataset):
            for name, fp_array in tqdm(grouped_data.items()):
                targets = d_targets[name]
                try:
                    assert fp_array.shape[0] == len(targets)
                    results = self.evaluate_input_list(input_list=fp_array,
                                                    targets=targets)
                    mol_results, conf_results = results
                    ranker_mol_results[name] = mol_results
                    ranker_conf_results[name] = conf_results
                except Exception as e:
                    print(f'Evaluation failed for {name}')
                    print(str(e))
                
            # import pdb;pdb.set_trace()
                
            with open(self.ranker_mol_results_path, 'wb') as f:
                pickle.dump(ranker_mol_results, f)
            with open(self.ranker_conf_results_path, 'wb') as f:
                pickle.dump(ranker_conf_results, f)


    def group_by_name(self,
                      subset,
                      mol_ids):
        grouped_data = defaultdict(list)
        d_targets = defaultdict(list)
        print('Grouping by name')
        for mol_id, data in tqdm(zip(mol_ids, subset)):
            fp_array, rmsd = data
            name = mol_id.split('_')[0]
            grouped_data[name].append(fp_array)
            d_targets[name].append(rmsd.item())
            
        if isinstance(subset.dataset, E3FPDataset):
            print('Joining E3FP arrays')
            for name, data_list in tqdm(grouped_data.items()):
                grouped_data[name] = np.vstack(data_list)
        return grouped_data, d_targets
            

        
if __name__ == '__main__':
    
    def get_model(experiment_name, data_split) :
        checkpoint_name = os.listdir(os.path.join('lightning_logs', experiment_name, 'checkpoints'))[0]
        checkpoint_path = os.path.join('lightning_logs', experiment_name, 'checkpoints', checkpoint_name)
        config = {"num_interactions": 6,
                        "cutoff": 10,
                        "lr":1e-5,
                        'batch_size': 256,
                        'data_split': data_split}
        model = SchNetModel.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
        return model
    
    data_split = MoleculeSplit()
    experiment_name = 'random_split_0'
    
    
    # dataset = PyGDataset()
    # subsets = dataset.get_split_subsets(data_split)
    # mol_ids, test_subset = subsets['test']
    model = get_model(experiment_name,
                      data_split)
    ranker = ModelRanker(model, use_cuda=False)
    re = RankerEvaluator(experiment_name)
    re.evaluate_library()
    # re.evaluate_rankers(rankers, 
    #                     subset=test_subset, 
    #                     mol_ids=mol_ids)
    
    
    # dataset = RDKitMolDataset()
    # subsets = dataset.get_split_subsets(data_split)
    # mol_ids, test_subset = subsets['test']
    # rankers = [
    #     PropertyRanker('delta_u'),
    #     PropertyRanker('delta_e_sol'),
    #     PropertyRanker('delta_e_hyd'),
    #     PropertyRanker('delta_e_hphi'),
    #     PropertyRanker('delta_e_rgyr')
    # ]
    # re = RankerEvaluator(experiment_name)
    # re.evaluate_rankers(rankers, 
    #                     subset=test_subset, 
    #                     mol_ids=mol_ids)