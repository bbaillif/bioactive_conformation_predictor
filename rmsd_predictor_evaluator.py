import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from sklearn.metrics import r2_score
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Batch

class RMSDPredictorEvaluator() :
    
    def __init__(self, 
                 model, 
                 results_dir='results/', 
                 show_individual_scatterplot=False) :
        self.model = model
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir) :
            os.mkdir(results_dir)
        self.show_individual_scatterplot = show_individual_scatterplot
    
    def evaluate(self, dataset, training_dataset=None) :
        
        self.model.eval()
        if torch.cuda.is_available() :
            self.model.to('cuda')
        
        #d_smiles_indices = self.get_dataset_smiles_indices(dataset)
        grouped_data = self.group_dataset_by_smiles(dataset)
        
        if training_dataset is not None :
            train_smiles = set([data.data_id for data in train_subset])
            train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]
            train_fps = [AllChem.GetMorganFingerprint(mol, 3) for mol in train_mols]
            self.max_sims = defaultdict(list)
        
        self.smiles = []
        self.results_df = pd.DataFrame()
        self.n_generated = {}
        self.targets = {}
        self.preds = {}
        self.losses = {}
        self.all_bioactive_ranks = {}
        self.bioactive_accuracies = {}
        self.generated_accuracies = {}
        self.ccdc_accuracies = {}
        self.n_heavy_atoms = {}
        self.n_rotatable_bonds = {}
        self.ef20s_pred = {}
        self.ef20s_ccdc = {}
        
        print('Starting evaluation')
        for smiles, smiles_data_list in grouped_data.items() :
        
            self.smiles.append(smiles)
            smiles_loader = DataLoader(smiles_data_list, batch_size=16)
            mol_targets = []
            mol_preds = []
            with torch.no_grad() :
                for batch in smiles_loader :
                    batch.to(self.model.device)

                    pred = self.model(batch)
                    mol_preds.extend(pred.detach().cpu().numpy().squeeze(1))

                    target = self.model._get_target(batch)
                    mol_targets.extend(target.cpu().numpy())

                self.preds[smiles] = mol_preds
                self.targets[smiles] = mol_targets

                losses = F.mse_loss(torch.tensor(mol_targets), torch.tensor(mol_preds), reduction='none')
                self.losses[smiles] = losses.tolist()
            
            if self.show_individual_scatterplot :
                plt.scatter(mol_targets, mol_preds)
                plt.title(f'Loss : {loss:.2f}')
                plt.xlabel('RMSD')
                plt.ylabel('Predicted RMSD')
                plt.show()
               
            mol_targets = np.array(mol_targets)
            mol_preds = np.array(mol_preds)
            
            is_bioactive = mol_targets == 0
            
            real_ranks = mol_targets.argsort().argsort()
            pred_ranks = mol_preds.argsort().argsort()
            
            bioactive_real_ranks = real_ranks[is_bioactive]
            bioactive_pred_ranks = pred_ranks[is_bioactive]
            
            accuracy = len(set(bioactive_real_ranks).intersection(set(bioactive_pred_ranks))) / is_bioactive.sum()

            self.all_bioactive_ranks[smiles] = bioactive_pred_ranks
            self.bioactive_accuracies[smiles] = accuracy

            mol = Chem.MolFromSmiles(smiles_data_list[0].data_id)
            
            if training_dataset is not None :
                max_sim = self.get_max_sim_to_train_dataset(mol, train_fps)
                self.max_sims[smiles] = max_sim

            self.n_heavy_atoms[smiles] = mol.GetNumHeavyAtoms()
            self.n_rotatable_bonds[smiles] = CalcNumRotatableBonds(mol)
        
            mol_targets_generated = mol_targets[~is_bioactive]
            mol_preds_generated = mol_preds[~is_bioactive]
            
            self.n_generated[smiles] = len(mol_targets_generated)

            generated_real_order = mol_targets_generated.argsort()
            generated_pred_order = mol_preds_generated.argsort()
            generated_real_rank = mol_targets_generated.argsort().argsort()
            if generated_real_rank.shape[0] and generated_pred_order.shape[0] : # apparently there are cases where there is a shape 0 with size 0
                self.ccdc_accuracies[smiles] = generated_real_rank[0] == 0 # we check that the conformation with the lowest bioactive rmsd is the first one
                self.generated_accuracies[smiles] = generated_real_order[0] == generated_pred_order[0] # Is the predicted conf id the one with lowest rmsd
            n_generated = len(mol_targets_generated)
            top10_limit = int(n_generated * 10 / 100)
            top20_limit = int(n_generated * 20 / 100)
            if top10_limit != 0 and top20_limit != 0 :
                actives = generated_real_order[:top10_limit]
                active_rate_in_generated = len(actives) / n_generated

                active_rate_in_top20_predicted = len([i for i in generated_pred_order[:top20_limit] if i in actives]) / top20_limit
                ef20_pred = active_rate_in_top20_predicted / active_rate_in_generated
                self.ef20s_pred[smiles] = ef20_pred

                active_rate_in_top20_ccdc = len([i for i in range(top20_limit) if i in actives]) / top20_limit
                ef20_ccdc = active_rate_in_top20_ccdc / active_rate_in_generated
                self.ef20s_ccdc[smiles] = ef20_ccdc
        
    def evaluation_report(self, 
                          experiment_name,
                          training_dataset=None) :
        
        for smiles in self.smiles :
            self.results_df.loc[smiles, 'all_targets'] = str(self.targets[smiles])
            self.results_df.loc[smiles, 'all_preds'] = str(self.preds[smiles])
            self.results_df.loc[smiles, 'all_losses'] = str(self.losses[smiles])
            self.results_df.loc[smiles, 'mean_loss'] = np.mean(self.losses[smiles])
            self.results_df.loc[smiles, 'all_bioactive_rank'] = str(self.all_bioactive_ranks[smiles])
            self.results_df.loc[smiles, 'median_bioactive_rank'] = np.median(self.all_bioactive_ranks[smiles])
            self.results_df.loc[smiles, 'bioactive_accuracy'] = self.bioactive_accuracies[smiles]
            self.results_df.loc[smiles, 'n_bioactive'] = len(self.all_bioactive_ranks[smiles])
            self.results_df.loc[smiles, 'n_generated'] = self.n_generated[smiles]
            if smiles in self.generated_accuracies :
                self.results_df.loc[smiles, 'generated_accuracy'] = self.generated_accuracies[smiles]
                self.results_df.loc[smiles, 'ccdc_accuracy'] = self.ccdc_accuracies[smiles]
            self.results_df.loc[smiles, 'n_heavy_atoms'] = self.n_heavy_atoms[smiles]
            self.results_df.loc[smiles, 'n_rotatable_bonds'] = self.n_rotatable_bonds[smiles]
            if smiles in self.ef20s_pred :
                self.results_df.loc[smiles, 'ef20_pred'] = self.ef20s_pred[smiles]
                self.results_df.loc[smiles, 'ef20_ccdc'] = self.ef20s_ccdc[smiles]
        
        all_losses = [loss for smiles, losses in self.losses.items() for loss in losses]
        mean_loss = np.mean(all_losses)
        self.results_df.loc['Micro', 'mean_loss'] = mean_loss
        
        flatten_targets =  [target for smiles, targets in self.targets.items() for target in targets]
        flatten_preds =  [pred for smiles, preds in self.preds.items() for pred in preds]
        r2 = r2_score(flatten_targets, flatten_preds)
        
#         plt.figure(figsize=(10,10))
#         sns.kdeplot(x=flatten_targets, y=flatten_preds, fill=True)
#         #plt.scatter(flatten_targets, flatten_preds)
#         plt.title(f'R2 : {r2:.2f}')
#         plt.xlabel('RMSD')
#         plt.ylabel('Predicted RMSD')
#         plt.plot([0, 5], [0, 5], c='r')
#         plt.show()
        
        mean_bioactive_accuracy = np.mean([acc for smiles, acc in self.bioactive_accuracies.items()])
        self.results_df.loc['Micro', 'bioactive_accuracy'] = mean_bioactive_accuracy
        
        bioactive_ranks = [rank for smiles, ranks in self.all_bioactive_ranks.items() for rank in ranks]
        median_bioactive_rank = np.median(bioactive_ranks)
        self.results_df.loc['Micro', 'median_bioactive_rank'] = median_bioactive_rank
        
#         plt.figure(figsize=(7, 5))
#         plt.hist(bioactive_ranks, bins=100)
#         plt.xlabel('Rank')
#         plt.ylabel('Count')
#         plt.title('Distribution of predicted ranks of bioactive conformations')
#         #plt.savefig('Histogram_ranks', dpi=300)
#         plt.show()
        
        if training_dataset is not None :
            for smiles in self.smiles :
                plt.scatter(self.bioactive_accuracies[smiles], self.max_sims[smiles])
            plt.xlabel('Ranking accuracy')
            plt.ylabel('Maximum similarity to training set')
            plt.show()

            for smiles in self.smiles :
                plt.scatter(x=self.losses[smiles], y=self.max_sims[smiles])
            plt.xlabel('RMSD loss')
            plt.ylabel('Maximum similarity to training set')
            plt.show()
        
#         ls = [np.mean(self.losses[smiles]) for smiles in self.smiles]
#         ns = [self.n_heavy_atoms[smiles] for smiles in self.smiles]
#         plt.scatter(x=ls, y=ns)
#         plt.xlabel('Mean RMSD loss')
#         plt.ylabel('Number of heavy atom')
#         plt.show()
        
        mean_generated_accuracy = np.mean([acc for smiles, acc in self.generated_accuracies.items()])
        self.results_df.loc['Micro', 'generated_accuracy'] = mean_generated_accuracy
        
        mean_ccdc_accuracy = np.mean([acc for smiles, acc in self.ccdc_accuracies.items()])
        self.results_df.loc['Micro', 'ccdc_accuracy'] = mean_ccdc_accuracy
        
        mean_ef20_pred = np.mean([ef for smiles, ef in self.ef20s_pred.items()])
        self.results_df.loc['Micro', 'ef20_pred'] = mean_ef20_pred
        
        mean_ef20_ccdc = np.mean([ef for smiles, ef in self.ef20s_ccdc.items()])
        self.results_df.loc['Micro', 'ef20_ccdc'] = mean_ef20_ccdc
        
        results_csv_path = os.path.join(self.results_dir, experiment_name)
        self.results_df.to_csv(results_csv_path)
        
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