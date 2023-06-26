import time
import copy

from docking.pdbbind_docking import PDBbindDocking
from data.split import RandomSplit, ScaffoldSplit
from conf_ensemble import ConfEnsembleLibrary
from rankers import (NoRankingRanker, 
                     RandomRanker, 
                     EnergyRanker, 
                     ModelRanker,
                     PropertyRanker)
from rankers.tfd_ranker_sim import TFD2SimRefMCSRanker
from model import (AtomisticNNModel,
                   SchNetModel, 
                   DimeNetModel, 
                   ComENetModel)
from typing import List

pdbbind_docking = PDBbindDocking()
    
random_splits = [RandomSplit(split_i=i) for i in range(5)]
scaffolds_splits = [ScaffoldSplit(split_i=i) for i in range(5)]
data_splits = random_splits + scaffolds_splits

# data_splits = [RandomSplit()]

test_pdb_ids = []
for data_split in data_splits:
    test_pdb_ids.extend(data_split.get_pdb_ids(subset_name='test'))
test_pdb_ids = set(test_pdb_ids)

pdbbind_docking.dock_mol_pool(test_pdb_ids=test_pdb_ids)

master_cel = ConfEnsembleLibrary()

start_time = time.time()
for data_split in data_splits:
    
    train_smiles = data_split.get_smiles('train')
    train_cel = copy.deepcopy(master_cel)
    train_cel.select_smiles_list(smiles_list=train_smiles)
    tfd_ranker = TFD2SimRefMCSRanker(cel=train_cel)
    
    rankers = [NoRankingRanker(),
               RandomRanker(),
               EnergyRanker(),
               
               PropertyRanker(descriptor_name='Gold.PLP.Fitness',
                              ascending=False),
               tfd_ranker]
    
    model_classes: List[AtomisticNNModel] = [SchNetModel, 
                                             DimeNetModel, 
                                             ComENetModel]
    # rankers = []
    # model_classes: List[AtomisticNNModel] = [DimeNetModel, ComENetModel]
    for model_class in model_classes:
        model = model_class.get_model_for_data_split(data_split)
        rankers.append(ModelRanker(model, use_cuda=True))
    
    # import pdb;pdb.set_trace()
    
    test_pdb_ids = data_split.get_pdb_ids(subset_name='test')
    # test_pdb_ids = ['2h9t']
    
    pdbbind_docking.docking_analysis_pool(rankers=rankers,
                                        pdb_ids=test_pdb_ids,
                                        data_split=data_split,
                                        single=True)
    
    for task in ['all', 'hard', 'easy']:
        print(task)
        print('All results')
        pdbbind_docking.analysis_report(data_split=data_split,
                                      rankers=rankers,
                                    task=task,
                                    only_good_docking=False)
        print('Only good docking results')
        pdbbind_docking.analysis_report(data_split=data_split,
                                      rankers=rankers,
                                      task=task,
                                      only_good_docking=True)
        
# runtime = time.time() - start_time
# print(f'{runtime} seconds runtime')