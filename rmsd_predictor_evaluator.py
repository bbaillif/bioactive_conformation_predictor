import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
from litschnet import LitSchNet
from molsize_model import MolSizeModel
from numpy.random import default_rng
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment, CalcAUC, CalcROC
from scipy.stats import pearsonr, spearmanr

class RMSDPredictorEvaluator() :
    
    def __init__(self, 
                 model: LitSchNet, 
                 evaluation_name: str,
                 results_dir: str='results/', 
                 active_ratio: float=0.1,
                 show_individual_scatterplot: bool=False,
                 training_smiles: list=None,
                 training_pdb_ids: list=None) :
        
        self.model = model
        self.evaluation_name = evaluation_name
        self.results_dir = results_dir
        self.active_ratio = active_ratio
        self.show_individual_scatterplot = show_individual_scatterplot
        self.training_smiles = training_smiles
        self.training_pdb_ids = training_pdb_ids # useless currently, can be used for protein similarity to training set
        
        self.ef_step = 0.01
        self.ef_fractions = np.arange(0 + self.ef_step,
                                      1 + self.ef_step,
                                      self.ef_step)
        self.ef_fractions = np.around(self.ef_fractions, 2)
        
        self.rankers = ['random', 'energy', 'ccdc', 'model']
        
        # Define pathes
        if not os.path.exists(self.results_dir) :
            os.mkdir(self.results_dir)
        self.working_dir = os.path.join(self.results_dir, self.evaluation_name)
        if not os.path.exists(self.working_dir) :
            os.mkdir(self.working_dir)
        self.conf_results_path = os.path.join(self.working_dir, 'conf_results.p')
        self.mol_results_path = os.path.join(self.working_dir, 'mol_results.p')
            
        # To store results
        self.conf_results = defaultdict(dict)
        self.mol_results = defaultdict(dict)
        self.dataset_results = {}
        
        # For the random shuffle
        self.rng = default_rng()
    
    def evaluate(self, 
                 dataset,
                 overwrite=False) :
        
        if len(self.training_smiles) :
            print('Computing training set fingerprints')
            train_mols = [Chem.MolFromSmiles(smiles) for smiles in self.training_smiles]
            self.train_fps = [AllChem.GetMorganFingerprint(mol, 3, useChirality=True) for mol in train_mols]

        if (not os.path.exists(self.conf_results_path)) or overwrite:
            self.model.eval()
            if torch.cuda.is_available() :
                self.model.to('cuda')

            grouped_data = self.group_dataset_by_smiles(dataset)

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
        
        
    def evaluation_report(self, 
                          task: str='all') :
        self.task = task
        self.included_smiles = []
            
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
        self.regression_evaluation()
        self.bioactive_evaluation()
        self.ranking_evaluation()
        
        self.dataset_results_path = os.path.join(self.working_dir, 
                                                 f'dataset_results_{self.task}.p')
        with open(self.dataset_results_path, 'wb') as f:
            pickle.dump(self.dataset_results, f)
        
        
    def get_max_sim_to_train_dataset(self, mol) :
        assert self.train_fps is not None, 'You need training smiles input to compute similarity to training'
        test_fp = AllChem.GetMorganFingerprint(mol, 3, useChirality=True)
        sims = []
        for train_fp in self.train_fps :
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
            smiles = data.data_id.split('_')[0]
            d[smiles].append(data)
        return d
    
    
    def plot_distribution_bioactive_ranks(self, bioactive_ranks, suffix=None) :
        plt.figure(figsize=(7, 5))
        plt.hist(bioactive_ranks, bins=100)
        if suffix == 'normalized' :
            plt.xlabel('Normalized rank')
        else :
            plt.xlabel('Rank')
        plt.ylabel('Count')
        if suffix == 'normalized' :
            plt.title('Distribution of predicted bioactive conformations normalized ranks')
        else :
            plt.title('Distribution of predicted bioactive conformations ranks')
        if suffix is None :
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
        plt.plot([0, 3], [0, 3], c='r')
        plt.savefig(os.path.join(self.working_dir, f'regression.png'), dpi=300)
        #plt.show()
        plt.close()
        
        
    def mol_evaluation(self, smiles, smiles_data_list) :
        mol = Chem.MolFromSmiles(smiles)
        self.mol_results[smiles]['n_rotatable_bonds'] = CalcNumRotatableBonds(mol)
        self.mol_results[smiles]['n_heavy_atoms'] = mol.GetNumHeavyAtoms()
        self.mol_results[smiles]['max_sim_to_training'] = self.get_max_sim_to_train_dataset(mol)

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
                #plt.title(f'Loss : {loss:.2f}')
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

            if type(self.model) != MolSizeModel :
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
                if type(self.model) != MolSizeModel :
                    self.mol_results[smiles]['pearson_gen'] = pearsonr(generated_targets, generated_preds)[0]
                    self.mol_results[smiles]['spearman_gen'] = spearmanr(generated_targets, generated_preds)[0]
            self.mol_results[smiles]['rmse_gen'] = mean_squared_error(generated_targets, generated_preds, squared=False)

            # Ranking
            actives_i = np.argsort(generated_targets)[:int(len(generated_targets) * self.active_ratio)]
            activity = [True if i in actives_i else False for i in range(len(generated_targets))]
            self.conf_results[smiles]['generated_activity'] = activity
            preds_array = np.array(list(zip(generated_preds, activity))) # compatible with RDKit ranking metrics

            self.conf_results[smiles]['normalized_rank'] = {}
            self.conf_results[smiles]['normalized_rank']['model'] = generated_preds.argsort().argsort() / n_generated
            ranks_ccdc = np.array(range(n_generated))
            self.conf_results[smiles]['normalized_rank']['ccdc'] = ranks_ccdc / n_generated
            self.conf_results[smiles]['normalized_rank']['energy'] = generated_energies.argsort().argsort() / n_generated
            self.rng.shuffle(generated_preds)
            self.conf_results[smiles]['normalized_rank']['random'] = generated_preds.argsort().argsort() / n_generated

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

            for fraction in self.ef_fractions :
                self.mol_results[smiles][f'ef_{fraction}'] = {}
            self.mol_results[smiles]['bedroc'] = {}
            rankers = self.conf_results[smiles]['ranked_lists'].keys()
            include_ranking = True
            for ranker in rankers :
                for fraction in self.ef_fractions :
                    if include_ranking :
                        ranked_list = self.conf_results[smiles]['ranked_lists'][ranker]
                        if ranked_list[:, 1].sum() != 0 : # if not enough conf, there are no "bioactive"
                            ef_result = CalcEnrichment(ranked_list, 
                                                        col=1, 
                                                        fractions=[fraction])
                            if len(ef_result) :
                                ef = ef_result[0]
                            else :
                                ef = 1
                            self.mol_results[smiles][f'ef_{fraction}'][ranker] = ef
                        else :
                            include_ranking = False
                        
                if include_ranking :
                    self.mol_results[smiles]['bedroc'][ranker] = CalcBEDROC(self.conf_results[smiles]['ranked_lists'][ranker], col=1, alpha=20)
        
        
    def plot_distribution(self,
                          values,
                          name) :
        print(name)
        print(max(values))
        sns.displot(values, kde=True)
        plt.xlabel(name)
        plt.ylabel('Counts')
        fig_path = os.path.join(self.working_dir, 
                                f'distribution_{name}.png')
        plt.savefig(fig_path, dpi=300)
        #plt.show()
        plt.close()
       
       
    def similarity_bins(self,
                        sims,
                        values) :
        sims = np.array(sims)
        values = np.array(values)
        step = 0.10
        limits = [np.around(x, 2) for x in np.arange(0 + step, 1 + step, step)]
        sims_order = np.argsort(sims)
        sims = sims[sims_order]
        values = values[sims_order]
        
        current_limit_i = 0
        current_limit = limits[current_limit_i]
        group_names = []
        grouped_values = []
        current_values = []
        for sim, acc in zip(sims, values) :
            if sim <= current_limit :
                current_values.append(acc)
            else :
                group_names.append(current_limit)
                grouped_values.append(current_values)
                current_limit_i = current_limit_i + 1
                current_limit = limits[current_limit_i]
                current_values = []
        if len(current_values) :
            group_names.append(current_limit)
            grouped_values.append(current_values)
                
        df = pd.DataFrame(columns=['name', 'value'])
        for name, values in zip(group_names, grouped_values) :
            for value in values :
                row = {'name' : name, 'value' : value}
                series = pd.Series(row)
                df = df.append(series, ignore_index=True)
                
        return df
    
    
    def get_descriptor_full_name(self,
                                 descriptor) :
        if descriptor == 'n_heavy_atoms' :
            xlabel = 'Number of heavy atoms'
        elif descriptor == 'n_rotatable_bonds' :
            xlabel = 'Number of rotatable bonds'
        else :
            xlabel = descriptor
        return xlabel
        
    
    def plot_losses_to_training_similarity(self) :
        xlabel = 'Maximum similarity to training set'
        
        # RMSE ALL
        ylabel = 'Mean RMSE loss'
        losses = []
        max_sims = []
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            losses.append(mol_result['rmse_all'])
            max_sims.append(mol_result['max_sim_to_training'])
        df = pd.DataFrame(zip(max_sims, losses), 
                          columns=[xlabel, ylabel])
        sns.jointplot(data=df, x=xlabel, y=ylabel, kind='hist', bins=30, cbar=True)
        fig_path = os.path.join(self.working_dir, 
                                'loss_vs_training_sim_displot.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        #RMSE GEN
        ylabel = 'Generated conformations RMSE'
        losses = []
        max_sims = []
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            losses.append(mol_result['rmse_gen'])
            max_sims.append(mol_result['max_sim_to_training'])
        df = pd.DataFrame(zip(max_sims, losses), 
                          columns=[xlabel, ylabel])
        sns.jointplot(data=df, x=xlabel, y=ylabel, kind='hist', bins=30, cbar=True)
        fig_path = os.path.join(self.working_dir, 
                                'rmse_gen_vs_training_sim_displot.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        #RMSE BIO
        ylabel = 'Bioactive conformation ARMSD error'
        losses = []
        max_sims = []
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            losses.append(mol_result['rmse_bio'])
            max_sims.append(mol_result['max_sim_to_training'])
        df = pd.DataFrame(zip(max_sims, losses), 
                          columns=[xlabel, ylabel])
        sns.jointplot(data=df, x=xlabel, y=ylabel, kind='hist', bins=30, cbar=True)
        fig_path = os.path.join(self.working_dir, 
                                'rmse_bio_vs_training_sim_displot.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
    
    def plot_losses_to_molecule_descriptor(self,
                                           descriptor='n_heavy_atoms') :
        xlabel = self.get_descriptor_full_name(descriptor)
        ylabel = 'Mean RMSE loss'
        losses = []
        descriptor_values = []
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            losses.append(mol_result['rmse_all'])
            descriptor_values.append(mol_result[descriptor])
        df = pd.DataFrame(zip(descriptor_values, losses), 
                          columns=[xlabel, ylabel])
        
        sns.jointplot(data=df, x=xlabel, y=ylabel, kind='hist', bins=30, cbar=True)
        fig_path = os.path.join(self.working_dir, 
                                f'loss_vs_{descriptor}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        

    def plot_accuracy_to_training_similarity(self) :
        xlabel = 'Similarity to training set'
        ylabel = 'Ratio of molecules with bioactive ranked first'
        top1_bioactives = []
        max_sims = []
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            top1_bioactives.append(mol_result['top1_bioactive'])
            max_sims.append(mol_result['max_sim_to_training'])
        df = self.similarity_bins(sims=max_sims,
                                  values=top1_bioactives)
        df.columns = [xlabel, ylabel]
        sns.barplot(data=df, x=xlabel, y=ylabel)
        fig_path = os.path.join(self.working_dir, 
                                'bioactive_accuracy_vs_training_sim.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        
    def plot_accuracy_to_molecule_descriptor(self,
                                           descriptor='n_heavy_atoms') :
        xlabel = self.get_descriptor_full_name(descriptor)
        ylabel = 'Ratio of molecules with bioactive ranked first'
        top1_bioactives = []
        descriptor_values = []
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            top1_bioactives.append(mol_result['top1_bioactive'])
            descriptor_values.append(mol_result[descriptor])
        df = pd.DataFrame(zip(descriptor_values, top1_bioactives), 
                          columns=[xlabel, ylabel])
        
        sns.lineplot(data=df, x=xlabel, y=ylabel)
        fig_path = os.path.join(self.working_dir, 
                                f'bioactive_accuracy_vs_{descriptor}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        
    def get_rank_and_max_sims(self,
                            normalized=True) :
        
        min_bioactive_ranks = []
        max_sims = []
        
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            rank = mol_result['min_bioactive_ranks']
            if normalized :
                n_bioactive = mol_result['n_bioactive']
                n_generated = mol_result['n_generated']
                n_confs = n_bioactive + n_generated
                normalized_rank = rank / n_confs
                min_bioactive_ranks.append(normalized_rank)
            else :
                min_bioactive_ranks.append(rank)
            max_sims.append(mol_result['max_sim_to_training'])
            
        return max_sims, min_bioactive_ranks
    
    
    def get_rank_and_descriptors(self,
                                descriptor='n_heavy_atoms',
                                normalized=True) :
        
        descriptor_values = []
        min_bioactive_ranks = []
        
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            rank = mol_result['min_bioactive_ranks']
            if normalized :
                n_bioactive = mol_result['n_bioactive']
                n_generated = mol_result['n_generated']
                n_confs = n_bioactive + n_generated
                normalized_rank = rank / n_confs
                min_bioactive_ranks.append(normalized_rank)
            else :
                min_bioactive_ranks.append(rank)
            descriptor_values.append(mol_result[descriptor])
            
        return descriptor_values, min_bioactive_ranks
        
        
    def plot_rank_to_training_similarity(self,
                                         normalized=True) :
        xlabel = 'Maximum similarity to training set'
        if normalized :
            ylabel = 'Normalized rank of bioactive'
        else :
            ylabel = 'Rank of bioactive conformation'
        max_sims, min_bioactive_ranks = self.get_rank_and_max_sims(normalized=normalized)
        df = self.similarity_bins(sims=max_sims,
                                  values=min_bioactive_ranks)
        df.columns = [xlabel, ylabel]
        sns.barplot(data=df, x=xlabel, y=ylabel)
        fig_path = os.path.join(self.working_dir, 
                                'median_rank_vs_training_sim.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        
    def plot_rank_to_molecule_descriptor(self,
                                           descriptor='n_heavy_atoms',
                                           normalized=True) :
        
        xlabel = self.get_descriptor_full_name(descriptor)
        if normalized :
            ylabel = 'Normalized rank of bioactive'
        else :
            ylabel = 'Rank of bioactive conformation'
        descriptor_values, min_bioactive_ranks = self.get_rank_and_descriptors(descriptor=descriptor,
                                                                               normalized=normalized)
        df = pd.DataFrame(zip(descriptor_values, min_bioactive_ranks), 
                          columns=[xlabel, ylabel])
        
        sns.lineplot(data=df, x=xlabel, y=ylabel)
        fig_path = os.path.join(self.working_dir, 
                                f'median_rank_vs_{descriptor}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        
    def plot_ef_to_training_similarity(self,
                                       fraction=0.1) :
        
        xlabel = 'Maximum similarity to training set'
        ylabel = f'Enrichment factor at fraction {fraction}'
        efs = defaultdict(list)
        max_sims = []
        
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            include_ranking = True
            for ranker in self.rankers :
                if ranker in mol_result[f'ef_{fraction}'] and include_ranking :
                    efs[ranker].append(mol_result[f'ef_{fraction}'][ranker])
                else :
                    include_ranking = False
            if include_ranking :
                max_sims.append(mol_result['max_sim_to_training'])
            
        for ranker in self.rankers :
            df = self.similarity_bins(sims=max_sims,
                                    values=efs[ranker])
            df.columns = [xlabel, ylabel]
            sns.barplot(data=df, x=xlabel, y=ylabel)
            fig_path = os.path.join(self.working_dir, 
                                    f'ef10_vs_training_sim_{ranker}.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            
    def plot_ef_to_molecule_descriptor(self,
                                       descriptor='n_heavy_atoms',
                                       fraction=0.1) :
        xlabel = self.get_descriptor_full_name(descriptor)
        ylabel = f'Enrichment factor at fraction {fraction}'
        efs = defaultdict(list)
        descriptor_values = []
        
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            include_ranking = True
            for ranker in self.rankers :
                if ranker in mol_result[f'ef_{fraction}'] and include_ranking :
                    efs[ranker].append(mol_result[f'ef_{fraction}'][ranker])
                else :
                    include_ranking = False
            if include_ranking :
                descriptor_values.append(mol_result[descriptor])
            
        for ranker in self.rankers :
            df = pd.DataFrame(zip(descriptor_values, efs[ranker]), 
                          columns=[xlabel, ylabel])
            sns.lineplot(data=df, x=xlabel, y=ylabel)
            fig_path = os.path.join(self.working_dir, 
                                    f'ef10_vs_{descriptor}_{ranker}.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            
    def plot_efs(self) :
        
        xlabel = 'Fraction'
        ylabel = 'Enrichment factor'
        df = pd.DataFrame(columns=[xlabel, ylabel, 'ranker'])
        for ranker in self.rankers :
            xs = []
            ys = []
            for fraction in self.ef_fractions :
                for smiles in self.included_smiles :
                    mol_result = self.mol_results[smiles]
                    if ranker in mol_result[f'ef_{fraction}'] :
                        xs.append(fraction)
                        ys.append(mol_result[f'ef_{fraction}'][ranker])
            current_df = pd.DataFrame(zip(xs, ys), columns=[xlabel, ylabel])
            current_df['ranker'] = ranker
            df = df.append(current_df, ignore_index=True)
        #import pdb; pdb.set_trace()
        sns.lineplot(data=df, x=xlabel, y=ylabel, hue='ranker')
        plt.title(f'Enrichment factors of the top 10% closest to bioactive')
        fig_path = os.path.join(self.working_dir, 
                                    f'efs.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        df_path = os.path.join(self.working_dir, 
                               'ef_df.csv')
        df.to_csv(df_path)
        
        
    def regression_evaluation(self) :
        self.dataset_results['regression'] = {}
            
        if len(self.training_smiles) :
            self.plot_losses_to_training_similarity()
            
        for descriptor in ['n_heavy_atoms', 'n_rotatable_bonds'] :
            self.plot_losses_to_molecule_descriptor(descriptor=descriptor)
            
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
            #self.plot_distribution(values=metric_values, name=metric)
            self.dataset_results['regression']['Macro'][metric] = np.mean(metric_values)
     
            
    def bioactive_evaluation(self) :
        self.dataset_results['bioactive_accuracy'] = {}
            
        if len(self.training_smiles) :
            self.plot_accuracy_to_training_similarity()
            self.plot_rank_to_training_similarity()
            
        for descriptor in ['n_heavy_atoms', 'n_rotatable_bonds'] :
            self.plot_accuracy_to_molecule_descriptor(descriptor=descriptor)
            self.plot_rank_to_molecule_descriptor(descriptor=descriptor)
            
        all_bioactive_ranks = []
        for smiles, results_d in self.conf_results.items() :
            if smiles in self.included_smiles :
                all_bioactive_ranks.extend([rank for rank in results_d['bioactive_ranks']])
        
        all_min_bioactive_ranks = []
        all_min_normalized_bioactive_ranks = []
        top1_accuracies = []
        topn_accuracies = []
        for smiles, results_d in self.mol_results.items() :
            if smiles in self.included_smiles :
                min_bioactive_rank = results_d['min_bioactive_ranks']
                n_bioactive = results_d['n_bioactive']
                n_generated = results_d['n_generated']
                n_confs = n_bioactive + n_generated
                normalized_bioactive_rank = min_bioactive_rank / n_confs
                all_min_bioactive_ranks.append(min_bioactive_rank)
                all_min_normalized_bioactive_ranks.append(normalized_bioactive_rank)
                top1_accuracies.append(results_d['top1_bioactive'])
                topn_accuracies.append(results_d['topN_bioactive'])
        
        self.plot_distribution_bioactive_ranks(all_bioactive_ranks)
        
        q1, median, q3 = np.quantile(all_bioactive_ranks, [0.25, 0.5, 0.75])
        self.dataset_results['bioactive_accuracy']['q1_all_bioactive'] = q1
        self.dataset_results['bioactive_accuracy']['median_all_bioactive'] = median
        self.dataset_results['bioactive_accuracy']['q3_all_bioactive'] = q3

        self.plot_distribution_bioactive_ranks(all_min_bioactive_ranks, 
                                               suffix='min')
        
        q1, median, q3 = np.quantile(all_min_bioactive_ranks, [0.25, 0.5, 0.75])
        self.dataset_results['bioactive_accuracy']['q1_min_bioactive'] = q1
        self.dataset_results['bioactive_accuracy']['median_min_bioactive'] = median
        self.dataset_results['bioactive_accuracy']['q3_min_bioactive'] = q3
        
        self.plot_distribution_bioactive_ranks(all_min_normalized_bioactive_ranks,
                                               suffix='normalized')
        q1, median, q3 = np.quantile(all_min_normalized_bioactive_ranks, [0.25, 0.5, 0.75])
        self.dataset_results['bioactive_accuracy']['q1_normalized_bioactive'] = q1
        self.dataset_results['bioactive_accuracy']['median_normalized_bioactive'] = median
        self.dataset_results['bioactive_accuracy']['q3_normalized_bioactive'] = q3
        
        self.dataset_results['bioactive_accuracy']['mean_top1_accuracy'] = np.mean(top1_accuracies)
        self.dataset_results['bioactive_accuracy']['mean_topN_accuracy'] = np.mean(topn_accuracies)
        
        
        
    def ranking_evaluation(self) :
        
        self.plot_ef_to_training_similarity()
        for descriptor in ['n_heavy_atoms', 'n_rotatable_bonds'] :
            self.plot_ef_to_molecule_descriptor(descriptor=descriptor)
        self.plot_efs()
        
        self.dataset_results['ranking'] = {}
        for ranker in self.rankers :
            bedrocs = []
            efs = defaultdict(list)
            for smiles, results_d in self.mol_results.items() :
                if smiles in self.included_smiles :
                    if 'bedroc' in results_d :
                        if ranker in results_d['bedroc'] :
                            bedrocs.append(results_d['bedroc'][ranker])
                            for fraction in self.ef_fractions :
                                efs[fraction].append(results_d[f'ef_{fraction}'][ranker])
            self.dataset_results['ranking'][ranker] = {}
            self.dataset_results['ranking'][ranker]['bedroc'] = np.mean(bedrocs)
            for fraction in self.ef_fractions :
                self.dataset_results['ranking'][ranker][f'ef_{fraction}'] = np.mean(efs[fraction])
         
    