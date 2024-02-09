import os
import numpy as np
import pandas as pd
import copy

from tqdm import tqdm
from typing import Tuple, Dict, Sequence
from bioconfpred.conf_ensemble import (ConfEnsembleLibrary, 
                                       ConfEnsemble)
from bioconfpred.evaluator import RankerEvaluator
from bioconfpred.model import (AtomisticNNModel,
                    SchNetModel, 
                   DimeNetModel, 
                   ComENetModel)
from bioconfpred.data.split import RandomSplit, ScaffoldSplit
from bioconfpred.ranker import (ConfRanker,
                                NoRankingRanker,
                                RandomRanker,
                                EnergyRanker,
                                SASARanker,
                                RGyrRanker,
                                ModelRanker,
                                TFD2SimRefMCSRanker)
from typing import List
from bioconfpred.params import (BIO_CONF_DIRPATH, 
                                GEN_CONF_DIRPATH,
                                RMSD_DIRPATH)

# PDB ID based
def get_ce_and_targets(filename: str, 
                       pdb_ids: List[str]
                       ) -> Tuple[Dict[str, ConfEnsemble], Dict[str, Sequence]]:
    pdb_filepath = os.path.join(BIO_CONF_DIRPATH, 
                                filename)
    gen_filepath = os.path.join(GEN_CONF_DIRPATH,
                                filename)
    pdb_ce = ConfEnsemble.from_file(pdb_filepath, 
                                    renumber_atoms=False)
    gen_ce = ConfEnsemble.from_file(gen_filepath, 
                                    renumber_atoms=False)
    
    pdb_idx_to_id = {}
    confs = [conf for conf in pdb_ce.mol.GetConformers()]
    for i, conf in enumerate(confs):
        pdb_id = conf.GetProp('PDB_ID')
        if pdb_id in pdb_ids:
            pdb_idx_to_id[i] = pdb_id
        else:
            pdb_ce.mol.RemoveConformer(conf.GetId())
            
    rmsd_filepath = os.path.join(RMSD_DIRPATH, 
                                 filename.replace('.sdf', '.npy'))
    rmsds = np.load(rmsd_filepath)
    gen_ce_rmsds = {}
    gen_ces = {}
    for i, pdb_id in pdb_idx_to_id.items():
        gen_ce_rmsds[pdb_id] = rmsds[:, i]
        gen_ces[pdb_id] = gen_ce
    return gen_ces, gen_ce_rmsds


cel_df = pd.read_csv(os.path.join(BIO_CONF_DIRPATH,
                                  'ensemble_names.csv'))
pdbbind_df = pd.read_csv(os.path.join(BIO_CONF_DIRPATH,
                                      'pdb_df.csv'))

master_cel = ConfEnsembleLibrary()

common_rankers = [RandomRanker(),
                  NoRankingRanker(), 
                  EnergyRanker(),
                  SASARanker(),
                  RGyrRanker(),
                  ]

split_types = ['random', 'scaffold']
split_is = range(5)

model_classes: List[AtomisticNNModel] = [SchNetModel, 
                                    DimeNetModel, 
                                    ComENetModel]

for split_type in split_types:
    
    for split_i in split_is:
        
        base_experiment_name = f'{split_type}_split_{split_i}'
        print(base_experiment_name)
        
        if split_type == 'random':
            data_split = RandomSplit(split_i=split_i)
        elif split_type == 'scaffold' :
            data_split = ScaffoldSplit(split_i=split_i)
        
        train_smiles = data_split.get_smiles('train')
        
        test_pdbs = data_split.get_pdb_ids('test')
        pdbbind_test_df = pdbbind_df[pdbbind_df['pdb_id'].isin(test_pdbs)]
        cel_df_test = cel_df.merge(pdbbind_test_df, 
                                   left_on='ensemble_name',
                                   right_on='ligand_name')
        
        ce_dict = {}
        d_targets = {}
        unique_filenames = cel_df_test['filename'].unique()
        for filename in tqdm(unique_filenames):
            try:
                gen_ces, targets = get_ce_and_targets(filename,
                                                      test_pdbs)
                ce_dict.update(gen_ces)
                d_targets.update(targets)
            except Exception as e:
                print(f'Processing failed for {filename}')
                print(str(e))
        cel = ConfEnsembleLibrary.from_ce_dict(ce_dict, cel_name='test_random_0')
        
        train_cel = copy.deepcopy(master_cel)
        train_cel.select_smiles_list(smiles_list=train_smiles)
        tfd_ranker = TFD2SimRefMCSRanker(cel=train_cel)
        
        split_rankers: List[ConfRanker] = common_rankers + [tfd_ranker]
        # split_rankers = [tfd_ranker]
        
        for ranker in split_rankers:
            print(ranker.name)
            evaluator = RankerEvaluator(ranker, evaluation_name=base_experiment_name)
            evaluator.evaluate_library(cel, d_targets)
            
        for model_class in model_classes:
            model = model_class.get_model_for_data_split(data_split=data_split)
            ranker = ModelRanker(model, use_cuda=True)
            evaluator = RankerEvaluator(ranker, evaluation_name=base_experiment_name)
            evaluator.evaluate_library(cel, d_targets)