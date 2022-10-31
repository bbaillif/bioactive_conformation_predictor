from unittest import result
import numpy as np
import os
import pickle

from tqdm import tqdm
from collections import defaultdict
from data.pyg_dataset import PyGDataset
from data.rdkit_mol_dataset import RDKitMolDataset
from model.bioschnet import BioSchNet
from data.data_split import MoleculeSplit
from rankers.conf_ranker import (ConfRanker, 
                     BioSchNetConfRanker, 
                     E3FPConfRanker, 
                     DescriptorConfRanker)
from typing import Sequence, List, Dict, Any
from torch.utils.data import Subset, Dataset
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint, CalcNumRotatableBonds
from rdkit.ML.Scoring.Scoring import CalcEnrichment, CalcBEDROC

class RankerEvaluator():
    
    def __init__(self, 
                 results_dir: str = '/home/bb596/hdd/pdbbind_bioactive/results/', 
                 bioactive_threshold: float = 1.0,
                 non_bioactive_threshold: float = 2.5,
                 training_smiles: list = None) :
        
        self.results_dir = results_dir
        self.bioactive_threshold = bioactive_threshold
        self.non_bioactive_threshold = non_bioactive_threshold
        self.training_smiles = training_smiles
        
        if self.training_smiles is not None :
            print('Computing training set fingerprints')
            train_mols = [Chem.MolFromSmiles(smiles) 
                          for smiles in self.training_smiles]
            self.train_fps = [GetMorganFingerprint(mol, 3, useChirality=True) 
                              for mol in train_mols]
        
        self.labels = ['bioactive_like', 'non_bioactive']
        
        self.ef_step = 0.01
        self.ef_fractions = np.arange(0 + self.ef_step,
                                      1 + self.ef_step,
                                      self.ef_step)
        self.ef_fractions = np.around(self.ef_fractions, 2)
    
    def evaluate_rankers(self,
                         evaluation_name: str,
                         rankers: List[ConfRanker],
                         subset: Subset,
                         mol_ids: Sequence[str]):
        for ranker in rankers:
            self.evaluate_ranker(evaluation_name, ranker, subset, mol_ids)
    
    def evaluate_ranker(self,
                        evaluation_name: str,
                        ranker: ConfRanker,
                        subset: Subset,
                        mol_ids: Sequence[str]):
        
        # Define pathes
        if not os.path.exists(self.results_dir) :
            os.mkdir(self.results_dir)
        self.working_dir = os.path.join(self.results_dir, self.evaluation_name)
        if not os.path.exists(self.working_dir) :
            os.mkdir(self.working_dir)
        self.conf_results_path = os.path.join(self.working_dir, 'conf_results.p')
        self.mol_results_path = os.path.join(self.working_dir, 'mol_results.p')
        
        # To store results
        self.conf_results = {}
        self.mol_results = {}
        self.dataset_results = {}
        
        grouped_data = self.group_data_by_name(subset, mol_ids)
        self.subset = subset
        self.ranker = ranker
        all_mol_results = {}
        for name, data_list in tqdm(grouped_data.items()):
            mol_results = self.evaluate_mol(data_list)
            all_mol_results[name] = mol_results
            
        ranker_dir = os.path.join(self.working_dir, ranker.name)
        if not os.path.exists(ranker_dir):
            os.mkdir(ranker_dir)
        mol_results_path = os.path.join(ranker_dir, 'mol_results.p')
            
        with open(mol_results_path, 'wb') as f:
            pickle.dump(all_mol_results, f)
    
    def group_data_by_name(self,
                           dataset: Dataset,
                           mol_ids: Sequence[str]) -> Dict[str, Mol]:
        print('Grouping data by ligand name')
        d = defaultdict(list)
        for data, mol_id in zip(dataset, mol_ids) :
            name = mol_id.split('_')[0]
            d[name].append(data)
        return d
    
    def evaluate_mol(self,
                     data_list: List[Any]):
        
        # props to implement: nrotbonds, nheavyatoms, maxsimtotraining
        # n_bioactive, n_generated, 
        
        results = {}
        
        # Including bioactive confs
        input_list = self.ranker.get_input_list_from_data_list(data_list)
        output_list = self.ranker.get_output_list_from_data_list(data_list)
        values = self.ranker.get_values(input_list)
        ranks = self.ranker.get_ranks(values)
        rmsds = output_list
        rmsds = np.array(rmsds)
        
        is_bioactive = rmsds == 0
        
        if np.sum(is_bioactive) > 0:
            bioactive_results = self.get_bioactive_ranking(values,
                                                        ranks,
                                                        is_bioactive)
            results['bioactive'] = bioactive_results
        
        # Generated confs only
        output_list = np.array(output_list)
        generated_mask = output_list > 0
        input_list = [data 
                      for i, data in enumerate(input_list)
                      if i in np.argwhere(generated_mask)]
        output_list = output_list[generated_mask]
        values = self.ranker.get_values(input_list)
        ranks = self.ranker.get_ranks(values)
        rmsds = output_list
        rmsds = np.array(rmsds)
        
        bioactive_like_idxs = np.argwhere(rmsds <= self.bioactive_threshold)
        non_bioactive_idxs = np.argwhere(rmsds >= self.bioactive_threshold)
        d_mask_idxs = {'bioactive_like' : bioactive_like_idxs,
                        'non_bioactive' : non_bioactive_idxs}
        
        for label, mask_idxs in d_mask_idxs.items():
            n_labelled = mask_idxs.shape[0]
            if n_labelled > 0:
                mask = [True 
                        if i in mask_idxs 
                        else False 
                        for i in range(len(rmsds))]
                label_results = self.get_mask_ranking(values,
                                                      ranks, 
                                                      mask)
                results[label] = label_results
                
        return results
                
    def get_mask_ranking(self,
                         values: Sequence[float],
                         ranks: Sequence[int],
                         mask: Sequence[bool],
                         enrichment: bool = True):
        results = {}
        assert len(ranks) == len(mask)
        n_confs = len(ranks)
        n_masked = np.sum(mask)
             
        mask_ranks = ranks[mask]
        results['first_rank'] = float(mask_ranks[0])
        
        normalized_ranks = mask_ranks / n_confs
        results['normalized_first_rank'] = float(normalized_ranks[0])
        
        if enrichment:
            results['ef'] = {}
            results['normalized_ef'] = {}
            
            # transform into RDKit ranking metrics
            values = np.array(values)
            mask = np.array(mask)
            sorting = values.argsort()
            sorted_values = values[sorting]
            sorted_mask = mask[sorting]
            ranked_list = np.array(list(zip(sorted_values, sorted_mask)))
            
            max_ef =  1 / (n_masked / n_confs)
            for fraction in self.ef_fractions :
                ef_result = CalcEnrichment(ranked_list, 
                                            col=1, 
                                            fractions=[fraction])
                if not len(ef_result) :
                    ef = 1
                else :
                    ef = ef_result[0]
                normalized_ef = ef / max_ef
                results['ef'][fraction] = ef
                results['normalized_ef'][fraction] = normalized_ef
                
            bedroc = CalcBEDROC(ranked_list, col=1, alpha=20)
            results['bedroc'] = bedroc
        
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

        
if __name__ == '__main__':
    
    def get_model(experiment_name, data_split) :
        checkpoint_name = os.listdir(os.path.join('lightning_logs', experiment_name, 'checkpoints'))[0]
        checkpoint_path = os.path.join('lightning_logs', experiment_name, 'checkpoints', checkpoint_name)
        config = {"num_interactions": 6,
                        "cutoff": 10,
                        "lr":1e-5,
                        'batch_size': 256,
                        'data_split': data_split}
        model = BioSchNet.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
        return model
    
    data_split = MoleculeSplit()
    experiment_name = 'random_split_0'
    
    # dataset = PyGDataset()
    # subsets = dataset.get_split_subsets(data_split)
    # mol_ids, test_subset = subsets['test']
    # model = get_model(experiment_name,
    #                   data_split)
    # rankers = [
    #     BioSchNetConfRanker(model, use_cuda=False),
    # ]
    # re = RankerEvaluator(experiment_name)
    # re.evaluate_rankers(rankers, 
    #                     subset=test_subset, 
    #                     mol_ids=mol_ids)
    
    
    dataset = RDKitMolDataset()
    subsets = dataset.get_split_subsets(data_split)
    mol_ids, test_subset = subsets['test']
    rankers = [
        DescriptorConfRanker('delta_u'),
        DescriptorConfRanker('delta_e_sol'),
        DescriptorConfRanker('delta_e_hyd'),
        DescriptorConfRanker('delta_e_hphi'),
        DescriptorConfRanker('delta_e_rgyr')
    ]
    re = RankerEvaluator(experiment_name)
    re.evaluate_rankers(rankers, 
                        subset=test_subset, 
                        mol_ids=mol_ids)