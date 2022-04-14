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
from sklearn.metrics import r2_score, mean_squared_error, roc_curve
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
                 bioactive_threshold: float=1.0,
                 show_individual_scatterplot: bool=False,
                 training_smiles: list=None,
                 training_pdb_ids: list=None) :
        
        self.model = model
        self.evaluation_name = evaluation_name
        self.results_dir = results_dir
        self.bioactive_threshold = bioactive_threshold
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
        self.conf_results = {}
        self.mol_results = {}
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
        for smiles, mol_results in self.mol_results.items() :
            n_generated = mol_results['n_generated']
            has_generated = n_generated > 1
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
        if 'normalized' in suffix :
            plt.xlabel('Normalized rank')
        else :
            plt.xlabel('Rank')
        plt.ylabel('Count')
        if 'normalized' in suffix :
            plt.title('Distribution of predicted bioactive conformations normalized ranks')
        else :
            plt.title('Distribution of predicted bioactive conformations ranks')
        if suffix is None :
            fig_title = f'Bioactive_rank_distribution.png'
        else :
            fig_title = f'Bioactive_rank_distribution_{suffix}.png'
        plt.savefig(os.path.join(self.working_dir, fig_title), dpi=300, transparent=True)
        #plt.show()
        plt.close()
        
        
    def plot_regression(self, targets, preds) :
        plt.figure(figsize=(6, 6))
        plt.scatter(x=targets, y=preds, s=2, c='lightblue')
        sns.kdeplot(x=targets, y=preds, color='blue')
        r2 = r2_score(targets, preds)
        plt.title(f'R2 = {np.around(r2, 2)}')
        plt.xlabel('ARMSD')
        plt.ylabel('Predicted ARMSD')
        
        max_pred = np.max(preds)
        max_target = np.max(targets)
        max_value = max(max_pred, max_target)
        
        plt.plot([0, max_value], [0, max_value], c='g')
        plt.xlim(0, max_value + 0.2)
        plt.ylim(0, max_value + 0.2)
        plt.savefig(os.path.join(self.working_dir, f'regression.png'), dpi=300, transparent=True)
        #plt.show()
        plt.close()
        
        
    def mol_evaluation(self, smiles, smiles_data_list) :
        
        mol_results = {}
        conf_results = {}
        
        mol = Chem.MolFromSmiles(smiles)
        
        # Mol descriptors
        mol_results['n_rotatable_bonds'] = CalcNumRotatableBonds(mol)
        mol_results['n_heavy_atoms'] = mol.GetNumHeavyAtoms()
        mol_results['max_sim_to_training'] = self.get_max_sim_to_train_dataset(mol)

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
            rmses = torch.sqrt(losses)

        conf_results['preds'] = mol_preds
        conf_results['targets'] = mol_targets
        conf_results['losses'] = losses.tolist()
        conf_results['rmse'] = rmses.tolist()
        conf_results['energies'] = mol_energies
            
        mol_targets = np.array(mol_targets)
        mol_preds = np.array(mol_preds)
        mol_energies = np.array(mol_energies)
            
        n_conformations = len(mol_targets)
            
        # Bioactive stats
        is_bioactive = mol_targets == 0
        n_bioactive = is_bioactive.sum()
        mol_results['n_bioactive'] = int(n_bioactive)
        
        is_generated = ~is_bioactive
        n_generated = is_generated.sum()
        mol_results['n_generated'] = int(n_generated)
        
        # Molecule level evaluation
        if n_generated > 1 :

            mol_results['r2_all'] = r2_score(mol_targets, mol_preds)
            mol_results['rmse_all'] = mean_squared_error(mol_targets, mol_preds, squared=False)

            if type(self.model) != MolSizeModel :
                mol_results['pearson_all'] = pearsonr(mol_targets, mol_preds)[0]
                mol_results['spearman_all'] = spearmanr(mol_targets, mol_preds)[0]

            # Bioactive stats
            bioactive_preds = mol_preds[is_bioactive]
            conf_results['bioactive_preds'] = bioactive_preds
            try :
                mol_results['rmse_bio'] = mean_squared_error(np.zeros_like(bioactive_preds), bioactive_preds, squared=False) 
            except :
                import pdb;pdb.set_trace()

            # Generated stats
            generated_targets = mol_targets[is_generated]
            generated_preds = mol_preds[is_generated]
            generated_energies = mol_energies[is_generated]

            conf_results['generated_targets'] = generated_targets
            conf_results['generated_preds'] = generated_preds
            conf_results['generated_energies'] = generated_energies

            mol_results['r2_gen'] = r2_score(generated_targets, generated_preds)
            mol_results['rmse_gen'] = mean_squared_error(generated_targets, generated_preds, squared=False)
            if type(self.model) != MolSizeModel :
                mol_results['pearson_gen'] = pearsonr(generated_targets, generated_preds)[0]
                mol_results['spearman_gen'] = spearmanr(generated_targets, generated_preds)[0]

            # Bioactive conformations ranking
            mol_results['first_bioactive_rank'] = {}
            mol_results['normalized_first_bioactive_rank'] = {}
            bio_conf_rankers = {'model' : mol_preds,
                                'energy' : mol_energies,
                                'random' : np.random.randn(n_conformations)}
            bio_ranking_results = {}
            
            for ranker, values in bio_conf_rankers.items() :
                ranker_results = {}
                sorting = values.argsort()
                ranks = sorting.argsort()
                bioactive_ranks = ranks[is_bioactive]
                ranker_results['bioactive_ranks'] = bioactive_ranks.tolist()
                first_bioactive_rank = bioactive_ranks.min()
                mol_results['first_bioactive_rank'][ranker] = int(first_bioactive_rank)
                normalized_rank = first_bioactive_rank / n_conformations
                mol_results['normalized_first_bioactive_rank'][ranker] = float(normalized_rank)
            
                bio_ranking_results[ranker] = ranker_results
            
            conf_results['bioactive_ranking'] = bio_ranking_results

            # Generated conformations ranking
            actives_i = np.argwhere(generated_targets <= self.bioactive_threshold)
            if actives_i.shape[0] : # at least 1 active
                activity = [True if i in actives_i else False for i in range(len(generated_targets))]
                conf_results['generated_activity'] = activity
                
                n_actives = np.sum(activity)
                mol_results['n_actives'] = n_actives

                for fraction in self.ef_fractions :
                    mol_results[f'ef_{fraction}'] = {}
                    mol_results[f'normalized_ef_{fraction}'] = {}
                    
                mol_results['bedroc'] = {}
                gen_conf_rankers = {'model' : generated_preds,
                            'ccdc' : np.array(range(n_generated)),
                            'energy' : generated_energies,
                            'random' : np.random.randn(n_generated)}
                gen_ranking_results = {}
                
                for ranker, values in gen_conf_rankers.items() :
                    ranker_results = {}
                    sorting = values.argsort()
                    ranks = sorting.argsort()
                    ranker_results['ranks'] = ranks.tolist()
                    normalized_ranks = ranks / n_generated
                    ranker_results['normalized_ranks'] = normalized_ranks.tolist()
                    values_activity = np.array(list(zip(values, activity))) # compatible with RDKit ranking metrics
                    ranked_list = values_activity[sorting]
                    ranker_results['ranked_list'] = ranked_list
                    
                    # Enrichment of bioactive-like
                    efs = []
                    normalized_efs = []
                    max_ef =  1 / (n_actives / n_generated)
                    for fraction in self.ef_fractions :
                        ef_result = CalcEnrichment(ranked_list, 
                                                    col=1, 
                                                    fractions=[fraction])
                        if not len(ef_result) :
                            ef = 1
                        else :
                            ef = ef_result[0]
                        efs.append(ef)
                        normalized_efs.append(ef / max_ef)
                    
                    # efs = np.around(efs, decimals=3)
                    if len(efs) != len(self.ef_fractions) :
                        import pdb;pdb.set_trace()
                    for fraction, ef in zip(self.ef_fractions, efs) :
                        mol_results[f'ef_{fraction}'][ranker] = ef
                        
                    # normalized_efs = np.around(normalized_efs, decimals=3)
                    for fraction, normalized_ef in zip(self.ef_fractions, normalized_efs) :
                        mol_results[f'normalized_ef_{fraction}'][ranker] = normalized_ef
                            
                    mol_results['bedroc'][ranker] = CalcBEDROC(ranked_list, 
                                                               col=1, 
                                                               alpha=20)
                    
                    gen_ranking_results = ranker_results
                    
                conf_results['generated_ranking'] = gen_ranking_results
        
        self.mol_results[smiles] = mol_results
        self.conf_results[smiles] = conf_results
        
        
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
        plt.savefig(fig_path, dpi=300, transparent=True)
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
        
        pearson_coeff = np.around(pearsonr(df[xlabel].values, df[ylabel].values)[0], 
                            3)
        sns.jointplot(data=df, 
                      x=xlabel, 
                      y=ylabel, 
                      kind='reg', 
                      xlim=(0, 1.1),
                      scatter_kws={"s" : 3},
                      line_kws={"color" : 'red',
                                "lw" : 2})
        plt.suptitle(f'Pearson : {pearson_coeff}')
        fig_path = os.path.join(self.working_dir, 
                                'loss_vs_training_sim.png')
        plt.savefig(fig_path, dpi=300, transparent=True)
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
        
        pearson_coeff = np.around(pearsonr(df[xlabel].values, df[ylabel].values)[0], 
                            3)
        sns.jointplot(data=df, 
                      x=xlabel, 
                      y=ylabel, 
                      kind='reg', 
                      xlim=(0, 1.1),
                      scatter_kws={"s" : 3},
                      line_kws={"color" : 'red',
                                "lw" : 2})
        plt.suptitle(f'Pearson : {pearson_coeff}')
        fig_path = os.path.join(self.working_dir, 
                                'rmse_gen_vs_training_sim_displot.png')
        plt.savefig(fig_path, dpi=300, transparent=True)
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
        
        pearson_coeff = np.around(pearsonr(df[xlabel].values, df[ylabel].values)[0], 
                            3)
        sns.jointplot(data=df, 
                      x=xlabel, 
                      y=ylabel, 
                      kind='reg', 
                      xlim=(0, 1.1),
                      scatter_kws={"s" : 3},
                      line_kws={"color" : 'red',
                                "lw" : 2})
        plt.suptitle(f'Pearson : {pearson_coeff}')
        fig_path = os.path.join(self.working_dir, 
                                'rmse_bio_vs_training_sim_displot.png')
        plt.savefig(fig_path, dpi=300, transparent=True)
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
        
        pearson_coeff = np.around(pearsonr(df[xlabel].values, df[ylabel].values)[0], 
                            3)
        sns.jointplot(data=df, 
                      x=xlabel, 
                      y=ylabel, 
                      kind='reg', 
                      scatter_kws={"s" : 3},
                      line_kws={"color" : 'red',
                                "lw" : 2})
        plt.suptitle(f'Pearson : {pearson_coeff}')
        fig_path = os.path.join(self.working_dir, 
                                f'loss_vs_{descriptor}.png')
        plt.savefig(fig_path, dpi=300, transparent=True)
        plt.close()
        

    def plot_accuracy_to_training_similarity(self) :
        xlabel = 'Similarity to training set'
        ylabel = 'Ratio of molecules with bioactive ranked first'
        for ranker in ['model', 'energy', 'random'] :
            title = ranker
            top1_bioactives = []
            max_sims = []
            for smiles in self.included_smiles :
                mol_results = self.mol_results[smiles]
                top1_bioactives.append(mol_results['first_bioactive_rank'][ranker] == 0)
                max_sims.append(mol_results['max_sim_to_training'])
            df = self.similarity_bins(sims=max_sims,
                                    values=top1_bioactives)
            df.columns = [xlabel, ylabel]
            sns.barplot(data=df, x=xlabel, y=ylabel)
            fig_path = os.path.join(self.working_dir, 
                                    f'bioactive_accuracy_vs_training_sim_{ranker}.png')
            plt.title(title)
            plt.savefig(fig_path, dpi=300, transparent=True)
            plt.close()
        
        
    def plot_accuracy_to_molecule_descriptor(self,
                                             descriptor='n_heavy_atoms') :
        xlabel = self.get_descriptor_full_name(descriptor)
        ylabel = 'Ratio of molecules with bioactive ranked first'
        for ranker in ['model', 'energy', 'random'] :
            top1_bioactives = []
            descriptor_values = []
            for smiles in self.included_smiles :
                mol_result = self.mol_results[smiles]
                top1_bioactives.append(mol_result['first_bioactive_rank'][ranker] == 0)
                descriptor_values.append(mol_result[descriptor])
            df = pd.DataFrame(zip(descriptor_values, top1_bioactives), 
                            columns=[xlabel, ylabel])
            
            sns.lineplot(data=df, x=xlabel, y=ylabel)
            fig_path = os.path.join(self.working_dir, 
                                    f'bioactive_accuracy_vs_{descriptor}_{ranker}.png')
            plt.savefig(fig_path, dpi=300, transparent=True)
            plt.close()
        
        
    def get_rank_and_max_sims(self,
                              ranker,
                              normalized=True) :
        
        min_bioactive_ranks = []
        max_sims = []
        
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            if normalized :
                rank = mol_result['normalized_first_bioactive_rank'][ranker]
            else :
                rank = mol_result['first_bioactive_rank'][ranker]
            min_bioactive_ranks.append(rank)
            max_sims.append(mol_result['max_sim_to_training'])
            
        return max_sims, min_bioactive_ranks
    
    
    def get_rank_and_descriptors(self,
                                 ranker,
                                 descriptor='n_heavy_atoms',
                                 normalized=True) :
        
        descriptor_values = []
        min_bioactive_ranks = []
        
        for smiles in self.included_smiles :
            mol_result = self.mol_results[smiles]
            if normalized :
                rank = mol_result['normalized_first_bioactive_rank'][ranker]
            else :
                rank = mol_result['first_bioactive_rank'][ranker]
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
        for ranker in ['model', 'energy', 'random'] :
            max_sims, min_bioactive_ranks = self.get_rank_and_max_sims(ranker=ranker,
                                                                       normalized=normalized)
            df = pd.DataFrame(zip(max_sims, min_bioactive_ranks), 
                            columns=[xlabel, ylabel])
            pearson_coeff = np.around(pearsonr(df[xlabel].values, df[ylabel].values)[0], 
                            3)
            sns.jointplot(data=df, 
                        x=xlabel, 
                        y=ylabel, 
                        kind='reg', 
                        xlim=(0, 1.1),
                        scatter_kws={"s" : 3},
                        line_kws={"color" : 'red',
                                    "lw" : 2})
            plt.suptitle(f'Pearson : {pearson_coeff}')
            fig_path = os.path.join(self.working_dir, 
                                    f'first_bio_rank_vs_training_sim_{ranker}.png')
            plt.savefig(fig_path, dpi=300, transparent=True)
            plt.close()
        
        
    def plot_rank_to_molecule_descriptor(self,
                                           descriptor='n_heavy_atoms',
                                           normalized=True) :
        
        xlabel = self.get_descriptor_full_name(descriptor)
        if normalized :
            ylabel = 'Normalized rank of bioactive'
        else :
            ylabel = 'Rank of bioactive conformation'
        for ranker in ['model', 'energy', 'random'] :
            descriptor_values, min_bioactive_ranks = self.get_rank_and_descriptors(ranker=ranker,
                                                                                    descriptor=descriptor,
                                                                                normalized=normalized)
            df = pd.DataFrame(zip(descriptor_values, min_bioactive_ranks), 
                            columns=[xlabel, ylabel])
            
            pearson_coeff = np.around(pearsonr(df[xlabel].values, df[ylabel].values)[0], 
                            3)
            sns.jointplot(data=df, 
                        x=xlabel, 
                        y=ylabel, 
                        kind='reg',
                        scatter_kws={"s" : 3},
                        line_kws={"color" : 'red',
                                    "lw" : 2})
            plt.suptitle(f'Pearson : {pearson_coeff}')
            fig_path = os.path.join(self.working_dir, 
                                    f'first_bio_rank_vs_{descriptor}_{ranker}.png')
            plt.savefig(fig_path, dpi=300, transparent=True)
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
                if f'ef_{fraction}' in mol_result :
                    if ranker in mol_result[f'ef_{fraction}'] and include_ranking :
                        efs[ranker].append(mol_result[f'ef_{fraction}'][ranker])
                    else :
                        include_ranking = False
                else :
                    include_ranking = False
            if include_ranking :
                max_sims.append(mol_result['max_sim_to_training'])
            
        for ranker in self.rankers :
            df = pd.DataFrame(zip(max_sims, efs[ranker]), 
                            columns=[xlabel, ylabel])
            pearson_coeff = np.around(pearsonr(df[xlabel].values, df[ylabel].values)[0], 
                            3)
            sns.jointplot(data=df, 
                        x=xlabel, 
                        y=ylabel, 
                        kind='reg', 
                        xlim=(0, 1.1),
                        scatter_kws={"s" : 3},
                        line_kws={"color" : 'red',
                                    "lw" : 2})
            plt.suptitle(f'Pearson : {pearson_coeff}')
            fig_path = os.path.join(self.working_dir, 
                                    f'ef10_vs_training_sim_{ranker}.png')
            plt.savefig(fig_path, dpi=300, transparent=True)
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
                if f'ef_{fraction}' in mol_result :
                    if ranker in mol_result[f'ef_{fraction}'] and include_ranking :
                        efs[ranker].append(mol_result[f'ef_{fraction}'][ranker])
                    else :
                        include_ranking = False
                else :
                    include_ranking = False
            if include_ranking :
                descriptor_values.append(mol_result[descriptor])
            
        for ranker in self.rankers :
            df = pd.DataFrame(zip(descriptor_values, efs[ranker]), 
                          columns=[xlabel, ylabel])
            pearson_coeff = np.around(pearsonr(df[xlabel].values, df[ylabel].values)[0], 
                            3)
            sns.jointplot(data=df, 
                        x=xlabel, 
                        y=ylabel, 
                        kind='reg',
                        scatter_kws={"s" : 3},
                        line_kws={"color" : 'red',
                                    "lw" : 2})
            plt.suptitle(f'Pearson : {pearson_coeff}')
            fig_path = os.path.join(self.working_dir, 
                                    f'ef10_vs_{descriptor}_{ranker}.png')
            plt.savefig(fig_path, dpi=300, transparent=True)
            plt.close()
            
            
    def plot_efs(self,
                 normalized=False) :
        
        xlabel = 'Fraction of ranked generated conformations'
        if normalized :
             ylabel = 'Normalized EF of bioactive-like conformations'
        else :
            ylabel = 'EF of bioactive-like conformations'
        df = pd.DataFrame(columns=[xlabel, ylabel, 'ranker'])
        for ranker in self.rankers :
            xs = []
            ys = []
            for fraction in self.ef_fractions :
                if normalized :
                    ef_entry = f'normalized_ef_{fraction}'
                else :
                    ef_entry = f'ef_{fraction}'
                for smiles in self.included_smiles :
                    mol_result = self.mol_results[smiles]
                    if ef_entry in mol_result :
                        if ranker in mol_result[ef_entry] :
                            xs.append(fraction)
                            ys.append(mol_result[ef_entry][ranker])
            current_df = pd.DataFrame(zip(xs, ys), columns=[xlabel, ylabel])
            current_df['ranker'] = ranker
            df = df.append(current_df, ignore_index=True)
        sns.lineplot(data=df, x=xlabel, y=ylabel, hue='ranker')
        if normalized :
            filename = 'normalized_efs.png'
        else :
            filename = 'efs.png'
        fig_path = os.path.join(self.working_dir, 
                                filename)
        plt.savefig(fig_path, dpi=300, transparent=True)
        plt.close()
        
        if normalized :
            filename = 'normalized_ef_df.csv'
        else :
            filename = 'ef_df.csv'
        df_path = os.path.join(self.working_dir, 
                               filename)
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
        for smiles, conf_results in self.conf_results.items() :
            if smiles in self.included_smiles :
                all_targets.extend([target for target in conf_results['targets']])
                all_preds.extend([target for target in conf_results['preds']])
                all_bioactive_preds.extend([target for target in conf_results['bioactive_preds']])
                if 'generated_targets' in conf_results :
                    all_generated_targets.extend([target for target in conf_results['generated_targets']])
                    all_generated_preds.extend([pred for pred in conf_results['generated_preds']])

        self.plot_regression(all_targets, all_preds)

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
            for smiles, mol_results in self.mol_results.items() :
                if smiles in self.included_smiles :
                    if metric in mol_results :
                        metric_values.append(mol_results[metric])
            #self.plot_distribution(values=metric_values, name=metric)
            self.dataset_results['regression']['Macro'][metric] = np.mean(metric_values)
     
            
    def bioactive_evaluation(self) :   
        
        if len(self.training_smiles) :
            self.plot_accuracy_to_training_similarity()
            self.plot_rank_to_training_similarity()
            
        for descriptor in ['n_heavy_atoms', 'n_rotatable_bonds'] :
            self.plot_accuracy_to_molecule_descriptor(descriptor=descriptor)
            self.plot_rank_to_molecule_descriptor(descriptor=descriptor)
            
        bioactive_ranking_results = {}
        for ranker in ['model', 'energy', 'random'] :
            bioactive_ranking_results[ranker] = {}
            all_bioactive_ranks = []
            for smiles, conf_results in self.conf_results.items() :
                if smiles in self.included_smiles :
                    ranker_results = conf_results['bioactive_ranking'][ranker]
                    bioactive_ranks = ranker_results['bioactive_ranks']
                    all_bioactive_ranks.extend([rank for rank in bioactive_ranks])
            
            all_min_bioactive_ranks = []
            all_min_normalized_bioactive_ranks = []
            top1_accuracies = []
            for smiles, mol_results in self.mol_results.items() :
                if smiles in self.included_smiles :
                    min_bioactive_rank = mol_results['first_bioactive_rank'][ranker]
                    normalized_bioactive_rank = mol_results['normalized_first_bioactive_rank'][ranker]
                    all_min_bioactive_ranks.append(min_bioactive_rank)
                    all_min_normalized_bioactive_ranks.append(normalized_bioactive_rank)
                    top1_accuracies.append(mol_results['first_bioactive_rank'][ranker] == 0)
            
            self.plot_distribution_bioactive_ranks(all_bioactive_ranks, 
                                                   suffix=f'{ranker}_all')
            
            q1, median, q3 = np.quantile(all_bioactive_ranks, [0.25, 0.5, 0.75])
            bioactive_ranking_results[ranker]['q1_all_bioactive'] = q1
            bioactive_ranking_results[ranker]['median_all_bioactive'] = median
            bioactive_ranking_results[ranker]['q3_all_bioactive'] = q3

            self.plot_distribution_bioactive_ranks(all_min_bioactive_ranks, 
                                                   suffix=f'{ranker}_first')
            
            q1, median, q3 = np.quantile(all_min_bioactive_ranks, [0.25, 0.5, 0.75])
            bioactive_ranking_results[ranker]['q1_min_bioactive'] = q1
            bioactive_ranking_results[ranker]['median_min_bioactive'] = median
            bioactive_ranking_results[ranker]['q3_min_bioactive'] = q3
            
            self.plot_distribution_bioactive_ranks(all_min_normalized_bioactive_ranks,
                                                   suffix=f'normalized_{ranker}')
            q1, median, q3 = np.quantile(all_min_normalized_bioactive_ranks, [0.25, 0.5, 0.75])
            bioactive_ranking_results[ranker]['q1_normalized_bioactive'] = q1
            bioactive_ranking_results[ranker]['median_normalized_bioactive'] = median
            bioactive_ranking_results[ranker]['q3_normalized_bioactive'] = q3
            
            bioactive_ranking_results[ranker]['mean_top1_accuracy'] = np.mean(top1_accuracies)
        
        self.dataset_results['bioactive_accuracy'] = bioactive_ranking_results
        
        
    def ranking_evaluation(self) :
        
        self.plot_ef_to_training_similarity()
        for descriptor in ['n_heavy_atoms', 'n_rotatable_bonds'] :
            self.plot_ef_to_molecule_descriptor(descriptor=descriptor)
        self.plot_efs()
        self.plot_efs(normalized=True)
        
        self.dataset_results['ranking'] = {}
        for ranker in self.rankers :
            bedrocs = []
            efs = defaultdict(list)
            normalized_efs = defaultdict(list)
            for smiles, mol_results in self.mol_results.items() :
                if smiles in self.included_smiles :
                    if 'bedroc' in mol_results :
                        if ranker in mol_results['bedroc'] :
                            bedrocs.append(mol_results['bedroc'][ranker])
                            for fraction in self.ef_fractions :
                                try :
                                    efs[fraction].append(mol_results[f'ef_{fraction}'][ranker])
                                    normalized_efs[fraction].append(mol_results[f'normalized_ef_{fraction}'][ranker])
                                except :
                                    import pdb;pdb.set_trace()
            self.dataset_results['ranking'][ranker] = {}
            self.dataset_results['ranking'][ranker]['bedroc'] = np.mean(bedrocs)
            for fraction in self.ef_fractions :
                self.dataset_results['ranking'][ranker][f'ef_{fraction}'] = np.mean(efs[fraction])
                self.dataset_results['ranking'][ranker][f'normalized_ef_{fraction}'] = np.mean(normalized_efs[fraction])
         
    