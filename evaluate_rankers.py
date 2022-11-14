import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple, Dict, Sequence
from conf_ensemble import ConfEnsembleLibrary, ConfEnsemble
from evaluator.ranker_evaluator import RankerEvaluator
from model import SchNetModel
from data.split import RandomSplit, ScaffoldSplit, UclusterSplit
from rankers.ccdc_ranker import CCDCRanker
from rankers.model_ranker import ModelRanker
from rankers.shuffle_ranker import ShuffleRanker
from rankers.property_ranker import PropertyRanker
from rankers.energy_ranker import EnergyRanker

def get_model_from_checkpoint(experiment_name, data_split) :
    checkpoint_dir = os.path.join('lightning_logs', experiment_name, 'checkpoints')
    checkpoint_name = os.listdir(checkpoint_dir)[0]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    config = {"num_interactions": 6,
                "cutoff": 10,
                "lr": 1e-5,
                'batch_size': 256,
                'data_split': data_split}
    model = SchNetModel.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
    return model

# Ligand conf ensemble based
# def get_ce_and_targets(filename, root, pdb_ids):
#     pdb_filepath = os.path.join(root, 'pdb_conf_ensembles_moe_all', filename)
#     gen_filepath = os.path.join(root, 'gen_conf_ensembles_moe_all', filename)
#     gen_ce = ConfEnsemble.from_file(gen_filepath)
#     pdb_ce = ConfEnsemble.from_file(pdb_filepath)
    
#     pdb_indices = []
#     confs = pdb_ce.mol.GetConformers()
#     for i, conf in enumerate(confs):
#         pdb_id = conf.GetProp('PDB_ID')
#         if not pdb_id in pdb_ids:
#             pdb_ce.mol.RemoveConformer(conf.GetId())
#         else:
#             pdb_indices.append(i)
    
#     rmsd_filepath = os.path.join(root, 'rmsds', filename.replace('.sdf', '.npy'))
#     rmsds = np.load(rmsd_filepath)
#     rmsds = rmsds[:, pdb_indices]
#     min_rmsds = rmsds.min(1).tolist()
    
#     gen_ce.add_mol(pdb_ce.mol, standardize=False)
#     min_rmsds.extend([0 for conf in pdb_ce.mol.GetConformers()])
#     return gen_ce, min_rmsds

# PDB ID based
def get_ce_and_targets(filename, root, pdb_ids) -> Tuple[Dict[str, ConfEnsemble], Dict[str, Sequence]]:
    pdb_filepath = os.path.join(root, 'pdb_conf_ensembles_moe_all', filename)
    gen_filepath = os.path.join(root, 'gen_conf_ensembles_moe_all', filename)
    pdb_ce = ConfEnsemble.from_file(pdb_filepath)
    gen_ce = ConfEnsemble.from_file(gen_filepath)
    
    pdb_idx_to_id = {}
    confs = pdb_ce.mol.GetConformers()
    for i, conf in enumerate(confs):
        pdb_id = conf.GetProp('PDB_ID')
        if pdb_id in pdb_ids:
            pdb_idx_to_id[i] = pdb_id
        else:
            pdb_ce.mol.RemoveConformer(conf.GetId())
            
    
    rmsd_filepath = os.path.join(root, 'rmsds', filename.replace('.sdf', '.npy'))
    rmsds = np.load(rmsd_filepath)
    gen_ce_rmsds = {}
    gen_ces = {}
    for i, pdb_id in pdb_idx_to_id.items():
        gen_ce_rmsds[pdb_id] = rmsds[:, i]
        gen_ces[pdb_id] = gen_ce
    return gen_ces, gen_ce_rmsds


root = '/home/bb596/hdd/pdbbind_bioactive/data'
cel_df = pd.read_csv(os.path.join(root, 
                                  'pdb_conf_ensembles', 
                                  'ensemble_names.csv'))
pdbbind_df = pd.read_csv(os.path.join(root,
                                      'pdbbind_df.csv'))

# split_types = ['random', 'scaffold', 'protein']
split_types = ['protein']
for split_type in split_types:
    
    for split_i in range(5):
        
        experiment_name = f'{split_type}_split_{split_i}'
        print(experiment_name)
        
        if split_type == 'random':
            data_split = RandomSplit(split_i=split_i)
        elif split_type == 'scaffold' :
            data_split = ScaffoldSplit(split_i=split_i)
        elif split_type == 'protein' :
            data_split = UclusterSplit(split_type=split_type,
                                       split_i=split_i)
        
        test_pdbs = data_split.get_pdb_ids('test')
        pdbbind_test_df = pdbbind_df[pdbbind_df['pdb_id'].isin(test_pdbs)]
        
        # test_smiles = data_split.get_smiles('test')
        # test_df = cel_df[cel_df['smiles'].isin(test_smiles)]
        
        ce_dict = {}
        d_targets = {}
        for i, row in tqdm(pdbbind_test_df.iterrows(), total=pdbbind_test_df.shape[0]):
            name = row['ligand_name']
            filename = cel_df[cel_df['ensemble_name'] == name]['filename'].values[0]
            try:
                gen_ces, targets = get_ce_and_targets(filename, root, test_pdbs)
                
                ce_dict.update(gen_ces)
                d_targets.update(targets)
            except Exception as e:
                print(f'Processing failed for {name}')
                print(str(e))
        cel = ConfEnsembleLibrary.from_ce_dict(ce_dict, cel_name='test_random_0')
        
        model = get_model_from_checkpoint(experiment_name, data_split)
        
        rankers = [
            ModelRanker(model, use_cuda=True),
            CCDCRanker(),
            ShuffleRanker(),
            EnergyRanker(),
            PropertyRanker('delta_u'),
        ]
        
        # descriptor_names = ['delta_u', 'delta_e_sol', 'delta_e_hyd',
        #                     'delta_e_hphi', 'delta_e_rgyr','E_sol', 'rgyr', 
        #                     'ASA_H', 'ASA_P', 'original_delta_e_sol']
        # descriptor_names = ['delta_u', 'delta_e_sol', 'E_sol']
        # rankers.extend([PropertyRanker(name) for name in descriptor_names])

        for ranker in rankers:
            print(ranker.name)
            evaluator = RankerEvaluator(ranker, evaluation_name=experiment_name)
            evaluator.evaluate_library(cel, d_targets)