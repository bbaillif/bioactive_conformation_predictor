import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader
from rdkit import DataStructs
from sklearn.metrics import r2_score
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Batch

class RMSDPredictorEvaluator() :
    
    def __init__(self, model, show_individual_scatterplot=False) :
        self.model = model
        self.show_individual_scatterplot = show_individual_scatterplot
        self.model.eval()
        if torch.cuda.is_available() :
            self.model.to('cuda')
    
    def evaluate(self, dataset) :
        
        self.evaluate_predictions(dataset)
        
    def evaluate_predictions(self, dataset, training_dataset=None) :
        
        #d_smiles_indices = self.get_dataset_smiles_indices(dataset)
        grouped_data = self.group_dataset_by_pdb_id(dataset)
        
        if training_dataset is not None :
            train_smiles = set([Chem.MolToSmiles(data.mol) for data in train_subset])
            train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]
            train_fps = [AllChem.GetMorganFingerprint(mol, 3) for mol in train_mols]
            max_sims = []
        
        #df = pd.DataFrame() Maybe TODO : return results as a table
        targets = [] 
        preds = []
        losses = []
        all_bioactive_ranks = []
        bioactive_accuracies = []
        generated_accuracies = []
        ccdc_accuracies = []
        n_heavy_atoms = []
        ef20s_pred = []
        ef20s_ccdc = []
        
        print('Starting evaluation')
        for pdb_id, pdb_id_data_list in tqdm(grouped_data.items()) :
        
            pdb_id_loader = DataLoader(pdb_id_data_list, batch_size=16)
            mol_targets = []
            mol_preds = []
            for batch in pdb_id_loader :
                batch.to(self.model.device)

                pred = self.model(batch)
                mol_preds.extend(pred.detach().cpu().numpy().squeeze(1))

                target = self.model._get_target(batch)
                mol_targets.extend(target.cpu().numpy())
        
            preds.append(mol_preds)
            targets.append(mol_targets)
        
            loss = F.mse_loss(torch.tensor(mol_targets), torch.tensor(mol_preds)).item()
            losses.append(loss)
            
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

            all_bioactive_ranks.extend(bioactive_pred_ranks)
            bioactive_accuracies.append(accuracy)

            mol = pdb_id_data_list[0].mol
            
            if training_dataset is not None :
                max_sim = self.get_max_sim_to_train_dataset(mol, train_fps)
                max_sims.append(max_sim)

            n_heavy_atoms.append(mol.GetNumHeavyAtoms())   
        
            mol_targets_generated = mol_targets[~is_bioactive]
            mol_preds_generated = mol_preds[~is_bioactive]

            generated_real_order = mol_targets_generated.argsort()
            generated_pred_order = mol_preds_generated.argsort()
            generated_real_rank = mol_targets_generated.argsort().argsort()
            if generated_real_rank.shape[0] and generated_pred_order.shape[0] : # apparently there are cases where there is a shape 0 with size 0
                ccdc_accuracies.append(generated_real_rank[0] == 0) # we check that the conformation with the lowest bioactive rmsd is the first one
                generated_accuracies.append(generated_real_order[0] == generated_pred_order[0]) # Is the predicted conf id the one with lowest rmsd
            n_generated = len(mol_targets_generated)
            top10_limit = int(n_generated * 10 / 100)
            top20_limit = int(n_generated * 20 / 100)
            if top10_limit != 0 and top20_limit != 0 :
                actives = generated_real_order[:top10_limit]
                active_rate_in_generated = len(actives) / n_generated

                active_rate_in_top20_predicted = len([i for i in generated_pred_order[:top20_limit] if i in actives]) / top20_limit
                ef20_pred = active_rate_in_top20_predicted / active_rate_in_generated
                ef20s_pred.append(ef20_pred)

                active_rate_in_top20_ccdc = len([i for i in range(top20_limit) if i in actives]) / top20_limit
                ef20_ccdc = active_rate_in_top20_ccdc / active_rate_in_generated
                ef20s_ccdc.append(ef20_ccdc)
        
        mean_loss = np.mean(losses)
        print(f'Mean loss : {mean_loss:.2f}')
        
        flatten_targets =  [item for sublist in targets for item in sublist]
        flatten_preds =  [item for sublist in preds for item in sublist]
        r2 = r2_score(flatten_targets, flatten_preds)
        
        plt.figure(figsize=(10,10))
        sns.kdeplot(x=flatten_targets, y=flatten_preds, fill=True)
        #plt.scatter(flatten_targets, flatten_preds)
        plt.title(f'R2 : {r2:.2f}')
        plt.xlabel('RMSD')
        plt.ylabel('Predicted RMSD')
        plt.plot([0, 5], [0, 5], c='r')
        plt.show()
        
        print(f'Bioactive ranking accuracy : {np.mean(bioactive_accuracies)}')
        print(f'Median rank : {np.median(all_bioactive_ranks)}')
        
        plt.figure(figsize=(7, 5))
        plt.hist(all_bioactive_ranks, bins=49)
        plt.xlabel('Rank')
        plt.ylabel('Count')
        plt.title('Distribution of predicted ranks of bioactive conformations')
        plt.savefig('Histogram_ranks', dpi=300)
        plt.show()
        
        if training_dataset is not None :
            plt.scatter(bioactive_accuracies, max_sims)
            plt.xlabel('Ranking accuracy')
            plt.ylabel('Maximum similarity to training set')
            plt.show()

            plt.scatter(x=losses, y=max_sims)
            plt.xlabel('RMSD loss')
            plt.ylabel('Maximum similarity to training set')
            plt.show()
        
        #plt.hist(n_heavy_atoms, losses)
        plt.scatter(losses, n_heavy_atoms)
        #plt.xlabel('RMSD loss')
        #plt.ylabel('Number of heavy atom')
        #plt.xlim(0, 2)
        plt.show()
        
        print(f'Generated ranking accuracy : {np.mean(generated_accuracies)}')
        print(f'CCDC ranking accuracy : {np.mean(ccdc_accuracies)}')
        print(f'EF20% prediction : {np.mean(ef20s_pred)}')
        print(f'EF20% CCDC : {np.mean(ef20s_ccdc)}')
        
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
    
    def group_dataset_by_pdb_id(self, dataset) :
        print('Grouping data by pdb_id')
        d = defaultdict(list)
        for data in dataset :
            pdb_id = data.mol.GetConformer().GetProp('PDB_ID')
            d[pdb_id].append(data)
        return d