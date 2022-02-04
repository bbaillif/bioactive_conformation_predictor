import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
import os
import copy
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Batch
from litschnet import LitSchNet
from numpy.random import default_rng
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment, CalcAUC, CalcROC
from scipy.stats import pearsonr, spearmanr

class RMSDPredictorEvaluator() :
    
    def __init__(self, 
                 model: LitSchNet, 
                 evaluation_name: str,
                 results_dir: str='results/', 
                 active_ratio: float=0.1,
                 show_individual_scatterplot:bool=False) :
        
        self.model = model
        
        self.evaluation_name = evaluation_name
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir) :
            os.mkdir(self.results_dir)
        self.working_dir = os.path.join(self.results_dir, self.evaluation_name)
        if not os.path.exists(self.working_dir) :
            os.mkdir(self.working_dir)
        self.conf_results_path = os.path.join(self.working_dir, 'conf_results.p')
        self.mol_results_path = os.path.join(self.working_dir, 'mol_results.p')
            
        self.conf_results = defaultdict(dict)
        self.mol_results = defaultdict(dict)
        self.dataset_results = {}
            
        self.active_ratio = 0.1
        self.show_individual_scatterplot = show_individual_scatterplot
        
        self.rng = default_rng()
    
    def evaluate(self, 
                 dataset,
                 training_dataset=None) :
        
        if not os.path.exists(self.conf_results_path):
            self.model.eval()
            if torch.cuda.is_available() :
                self.model.to('cuda')

            grouped_data = self.group_dataset_by_smiles(dataset)

            if training_dataset is not None :
                train_smiles = set([data.data_id for data in train_subset])
                train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]
                train_fps = [AllChem.GetMorganFingerprint(mol, 3) for mol in train_mols]
                self.max_sims = defaultdict(list)

            print('Starting evaluation')
            for smiles, smiles_data_list in tqdm(grouped_data.items()) :
                self.mol_evaluation(smiles, smiles_data_list)
                
            with open(self.mol_results_path, 'wb') as f:
                pickle.dump(self.mol_results, f)
            with open(self.conf_results_path, 'wb') as f:
                pickle.dump(self.conf_results, f)
                
        else :
            print(f'Evaluation already done for given experiment {self.evaluation_name}')
            print('Loading existing results')
            with open(self.mol_results_path, 'rb') as f:
                self.mol_results = pickle.load(f)
            with open(self.conf_results_path, 'rb') as f:
                self.conf_results = pickle.load(f)
        
        
    def evaluation_report(self, task: str='all') :
        self.task = task
        self.included_smiles = []
        
#         if training_dataset is not None :
#             for smiles in self.smiles :
#                 plt.scatter(self.bioactive_accuracies[smiles], self.max_sims[smiles])
#             plt.xlabel('Ranking accuracy')
#             plt.ylabel('Maximum similarity to training set')
#             plt.show()

#             for smiles in self.smiles :
#                 plt.scatter(x=self.losses[smiles], y=self.max_sims[smiles])
#             plt.xlabel('RMSD loss')
#             plt.ylabel('Maximum similarity to training set')
#             plt.show()
            
        # Define if the molecule will be included in dataset evaluation based on task
        for smiles, results_d in self.mol_results.items() :
            n_generated = results_d['n_generated']
            has_generated = n_generated > 0
            is_easy = n_generated < 100
            is_hard = n_generated == 100
            task_filter = (self.task == 'all') or (self.task == 'hard' and is_hard) or (self.task == 'easy' and is_easy)
            include_smiles = task_filter and has_generated
            if include_smiles :
                self.included_smiles.append(smiles)
            
        # Dataset level evaluation
        self.bioactive_evaluation()
        self.regression_evaluation()
        self.ranking_evaluation()
        
        self.dataset_results_path = os.path.join(self.working_dir, f'dataset_results_{self.task}.p')
        with open(self.dataset_results_path, 'wb') as f:
            pickle.dump(self.dataset_results, f)
        
        
    def get_max_sim_to_train_dataset(self, mol, train_fps) :
        test_fp = AllChem.GetMorganFingerprint(mol, 3)
        sims = []
        for train_fp in train_fps :
            dice_sim = DataStructs.TanimotoSimilarity(test_fp, train_fp)
            sims.append(dice_sim)
        max_sim = max(sims)
        return max_sim
    
    
    def get_dataset_smiles_indices(self, dataset) :
        d = defaultdict(list)
        for i, data in enumerate(dataset) :
            smiles = Chem.MolToSmiles(data.mol)
            d[smiles].append(i)
        return d
    
    
    def group_dataset_by_smiles(self, dataset) :
        print('Grouping data by smiles')
        d = defaultdict(list)
        for data in dataset :
            smiles = data.data_id
            d[smiles].append(data)
        return d
    
    
    def plot_distribution_bioactive_ranks(self, bioactive_ranks, suffix=None) :
        plt.figure(figsize=(7, 5))
        plt.hist(bioactive_ranks, bins=100)
        plt.xlabel('Rank')
        plt.ylabel('Count')
        plt.title('Distribution of predicted ranks of bioactive conformations')
        if not suffix is None :
            fig_title = f'Bioactive_rank_distribution.png'
        else :
            fig_title = f'Bioactive_rank_distribution_{suffix}.png'
        plt.savefig(os.path.join(self.working_dir, fig_title), dpi=300)
        #plt.show()
        plt.close()
        
        
    def plot_regression(self, targets, preds) :
        plt.figure(figsize=(8, 8))
        sns.kdeplot(x=targets, y=preds, fill=True)
        plt.title(f'Regression')
        plt.xlabel('RMSD')
        plt.ylabel('Predicted RMSD')
        plt.plot([0, 5], [0, 5], c='r')
        plt.savefig(os.path.join(self.working_dir, f'regression.png'), dpi=300)
        plt.close()
        
        
    def mol_evaluation(self, smiles, smiles_data_list) :
        mol = Chem.MolFromSmiles(smiles_data_list[0].data_id)
        self.mol_results[smiles]['n_rotatable_bonds'] = CalcNumRotatableBonds(mol)
        self.mol_results[smiles]['n_heavy_atoms'] = mol.GetNumHeavyAtoms()

#         if training_dataset is not None :
#             max_sim = self.get_max_sim_to_train_dataset(mol, train_fps)
#             self.max_sims[smiles] = max_sim

        # Make model predictions
        smiles_loader = DataLoader(smiles_data_list, batch_size=16)
        mol_targets = []
        mol_preds = []
        mol_energies = []
        with torch.no_grad() :
            for batch in smiles_loader :
                mol_energies.extend(batch.energy.numpy())
                mol_targets.extend(batch.rmsd.numpy())
                batch.to(self.model.device)

                pred = self.model(batch)
                mol_preds.extend(pred.detach().cpu().numpy().squeeze(1))

            losses = F.mse_loss(torch.tensor(mol_targets), torch.tensor(mol_preds), reduction='none')

        self.conf_results[smiles]['preds'] = mol_preds
        self.conf_results[smiles]['targets'] = mol_targets
        self.conf_results[smiles]['rmse'] = losses.tolist()
        self.conf_results[smiles]['energies'] = mol_energies
            
        mol_targets = np.array(mol_targets)
        mol_preds = np.array(mol_preds)
        mol_energies = np.array(mol_energies)
            
        # Bioactive stats
        is_bioactive = mol_targets == 0
        n_bioactive = is_bioactive.sum()
        self.mol_results[smiles]['n_bioactive'] = n_bioactive
        n_generated = (~is_bioactive).sum()
        self.mol_results[smiles]['n_generated'] = n_generated
        
        # Molecule level evaluation
        if n_generated > 0 :

            if self.show_individual_scatterplot :
                plt.scatter(mol_targets, mol_preds)
                plt.title(f'Loss : {loss:.2f}')
                plt.xlabel('RMSD')
                plt.ylabel('Predicted RMSD')
                plt.save()

            self.mol_results[smiles]['r2_all'] = r2_score(mol_targets, mol_preds)
            self.mol_results[smiles]['rmse_all'] = mean_squared_error(mol_targets, mol_preds, squared=False)

            pred_ranks = mol_preds.argsort().argsort()
            bioactive_pred_ranks = pred_ranks[is_bioactive]
            self.conf_results[smiles]['bioactive_ranks'] = bioactive_pred_ranks.tolist()
            self.mol_results[smiles]['min_bioactive_ranks'] = bioactive_pred_ranks.min()
            self.mol_results[smiles]['top1_bioactive'] = 0 in bioactive_pred_ranks
            topn_accuracy = len(set(range(n_bioactive)).intersection(set(bioactive_pred_ranks))) / n_bioactive
            self.mol_results[smiles]['topN_bioactive'] = topn_accuracy

            bioactive_preds = mol_preds[is_bioactive]
            self.conf_results[smiles]['bioactive_preds'] = bioactive_preds
            self.mol_results[smiles]['rmse_bio'] = mean_squared_error(np.zeros_like(bioactive_preds), bioactive_preds, squared=False) 

            self.mol_results[smiles]['pearson_all'] = pearsonr(mol_targets, mol_preds)[0]
            self.mol_results[smiles]['spearman_all'] = spearmanr(mol_targets, mol_preds)[0]

            # Generated stats
            generated_targets = mol_targets[~is_bioactive]
            generated_preds = mol_preds[~is_bioactive]
            generated_energies = mol_energies[~is_bioactive]

            self.conf_results[smiles]['generated_targets'] = generated_targets
            self.conf_results[smiles]['generated_preds'] = generated_preds
            self.conf_results[smiles]['generated_energies'] = generated_energies

            if n_generated > 1 :
                self.mol_results[smiles]['r2_gen'] = r2_score(generated_targets, generated_preds)
                self.mol_results[smiles]['pearson_gen'] = pearsonr(generated_targets, generated_preds)[0]
                self.mol_results[smiles]['spearman_gen'] = spearmanr(generated_targets, generated_preds)[0]
            self.mol_results[smiles]['rmse_gen'] = mean_squared_error(generated_targets, generated_preds, squared=False)

            # Ranking
            actives_i = np.argsort(generated_targets)[:int(len(generated_targets) * self.active_ratio)]
            activity = [True if i in actives_i else False for i in range(len(generated_targets))]
            self.conf_results[smiles]['generated_activity'] = activity
            preds_array = np.array(list(zip(generated_preds, activity))) # compatible with RDKit ranking metrics

            self.conf_results[smiles]['relative_rank'] = {}
            self.conf_results[smiles]['relative_rank']['model'] = generated_preds.argsort().argsort() / n_generated
            ranks_ccdc = np.array(range(n_generated))
            self.conf_results[smiles]['relative_rank']['ccdc'] = ranks_ccdc / n_generated
            self.conf_results[smiles]['relative_rank']['energy'] = generated_energies.argsort().argsort() / n_generated
            self.rng.shuffle(generated_preds)
            self.conf_results[smiles]['relative_rank']['random'] = generated_preds.argsort().argsort() / n_generated

            self.conf_results[smiles]['ranked_lists'] = {}

            self.conf_results[smiles]['ranked_lists']['ccdc'] = preds_array

            # Prediction ranking
            sorting = np.argsort(preds_array[:, 0])
            sorted_preds_array = preds_array[sorting]
            self.conf_results[smiles]['ranked_lists']['model'] = sorted_preds_array

            # Energy ranking
            preds_array = np.array(list(zip(generated_energies, activity)))
            sorting = np.argsort(preds_array[:, 0])
            sorted_preds_array = preds_array[sorting]
            self.conf_results[smiles]['ranked_lists']['energy'] = sorted_preds_array

            # Random ranking
            random_preds_array = copy.deepcopy(preds_array)
            self.rng.shuffle(random_preds_array)
            self.conf_results[smiles]['ranked_lists']['random'] = random_preds_array

            self.mol_results[smiles]['ef'] = {}
            self.mol_results[smiles]['bedroc'] = {}
            rankers = self.conf_results[smiles]['ranked_lists'].keys()
            for ranker in rankers :
                self.mol_results[smiles]['ef'][ranker] = CalcEnrichment(self.conf_results[smiles]['ranked_lists'][ranker], col=1, fractions=[0.2])[0]
                self.mol_results[smiles]['bedroc'][ranker] = CalcBEDROC(self.conf_results[smiles]['ranked_lists'][ranker], col=1, alpha=20)
        
        
    def regression_evaluation(self) :
        self.dataset_results['regression'] = {}
            
        all_targets = []
        all_preds = []
        all_bioactive_preds = []
        all_generated_targets = []
        all_generated_preds = []
        for smiles, results_d in self.conf_results.items() :
            if smiles in self.included_smiles :
                all_targets.extend([target for target in results_d['targets']])
                all_preds.extend([target for target in results_d['preds']])
                all_bioactive_preds.extend([target for target in results_d['bioactive_preds']])
                if 'generated_targets' in results_d :
                    all_generated_targets.extend([target for target in results_d['generated_targets']])
                    all_generated_preds.extend([pred for pred in results_d['generated_preds']])

        #self.plot_regression(all_targets, all_preds)

        # Micro
        self.dataset_results['regression']['Micro'] = {}
        self.dataset_results['regression']['Micro']['rmse_all'] = mean_squared_error(all_targets, all_preds, squared=False)
        self.dataset_results['regression']['Micro']['rmse_gen'] = mean_squared_error(all_generated_targets, all_generated_preds, squared=False)
        self.dataset_results['regression']['Micro']['rmse_bio'] = np.mean(all_bioactive_preds)
        self.dataset_results['regression']['Micro']['r2_all'] = r2_score(all_targets, all_preds)
        self.dataset_results['regression']['Micro']['r2_gen'] = r2_score(all_generated_targets, all_generated_preds)
        self.dataset_results['regression']['Micro']['pearson_all'] = pearsonr(all_targets, all_preds)[0]
        self.dataset_results['regression']['Micro']['pearson_gen'] = pearsonr(all_generated_targets, all_generated_preds)[0]
        self.dataset_results['regression']['Micro']['spearman_all'] = spearmanr(all_targets, all_preds)[0]
        self.dataset_results['regression']['Micro']['spearman_gen'] = spearmanr(all_generated_targets, all_generated_preds)[0]

        # Macro
        self.dataset_results['regression']['Macro'] = {}
        for metric in self.dataset_results['regression']['Micro'].keys() :
            metric_values = []
            for smiles, results_d in self.mol_results.items() :
                if smiles in self.included_smiles :
                    if metric in results_d :
                        metric_values.append(results_d[metric])
            self.dataset_results['regression']['Macro'][metric] = np.mean(metric_values)
            
            
    def bioactive_evaluation(self) :
        self.dataset_results['bioactive_accuracy'] = {}
            
        all_bioactive_ranks = []
        for smiles, results_d in self.conf_results.items() :
            if smiles in self.included_smiles :
                all_bioactive_ranks.extend([rank for rank in results_d['bioactive_ranks']])
        
        all_min_bioactive_ranks = []
        top1_accuracies = []
        topn_accuracies = []
        for smiles, results_d in self.mol_results.items() :
            if smiles in self.included_smiles :
                all_min_bioactive_ranks.append(results_d['min_bioactive_ranks'])
                top1_accuracies.append(results_d['top1_bioactive'])
                topn_accuracies.append(results_d['topN_bioactive'])
        
        self.plot_distribution_bioactive_ranks(all_bioactive_ranks)
        
        q1, median, q3 = np.quantile(all_bioactive_ranks, [0.25, 0.5, 0.75])
        self.dataset_results['bioactive_accuracy']['q1_all_bioactive'] = q1
        self.dataset_results['bioactive_accuracy']['median_all_bioactive'] = median
        self.dataset_results['bioactive_accuracy']['q3_all_bioactive'] = q3

        self.plot_distribution_bioactive_ranks(all_min_bioactive_ranks, suffix='min')
        
        q1, median, q3 = np.quantile(all_min_bioactive_ranks, [0.25, 0.5, 0.75])
        self.dataset_results['bioactive_accuracy']['q1_min_bioactive'] = q1
        self.dataset_results['bioactive_accuracy']['median_min_bioactive'] = median
        self.dataset_results['bioactive_accuracy']['q3_min_bioactive'] = q3

        self.dataset_results['bioactive_accuracy']['mean_top1_accuracy'] = np.mean(top1_accuracies)
        self.dataset_results['bioactive_accuracy']['mean_topN_accuracy'] = np.mean(topn_accuracies)
        
        
    def ranking_evaluation(self) :
        self.dataset_results['ranking'] = {}
        for ranker in ['random', 'energy', 'ccdc', 'model'] :
            bedrocs = []
            efs = []
            for smiles, results_d in self.mol_results.items() :
                if smiles in self.included_smiles :
                    if 'ef' in results_d :
                        bedrocs.append(results_d['bedroc'][ranker])
                        efs.append(results_d['ef'][ranker])
            self.dataset_results['ranking'][ranker] = {}
            self.dataset_results['ranking'][ranker]['bedroc'] = np.mean(bedrocs)
            self.dataset_results['ranking'][ranker]['ef'] = np.mean(efs)
         
    