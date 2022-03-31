import os
import time
import pickle
from rigid_docking import RigidDocking
from conf_ensemble_library import ConfEnsembleLibrary

# rigid_docking = RigidDocking(split_name='protein_similarity')
# rigid_docking = RigidDocking(split_name='random')

# rigid_docking.analysis_report(only_good_docking=False)

split_names = ['random', 'scaffold', 'protein_similarity']
# split_is = range(5)
split_is = [0]

conf_ensemble_library = ConfEnsembleLibrary()
conf_ensemble_library.load_metadata()

for split_name in split_names :
    print(split_name)
    for split_i in split_is :
        print(split_i)
        rigid_docking = RigidDocking(split_name=split_name,
                                    split_i=split_i)
        
        # test_pdb_ids = rigid_docking.get_test_pdb_ids()
        # rigid_docking.dock_molecule_pool(test_pdb_ids=test_pdb_ids)
        
        # start_time = time.time()  
        # rigid_docking.docking_analysis_pool()
        # runtime = time.time() - start_time
        # print(f'{runtime} seconds runtime')
        
        print('All results')
        rigid_docking.analysis_report(only_good_docking=False)
        
        print('Only good docking results')
        rigid_docking.analysis_report(only_good_docking=True)

split_name = 'random'
dataset = 'platinum'
split_i = 0
rigid_docking = RigidDocking(dataset=dataset,
                             split_name=split_name,
                             split_i=split_i)
        
# test_pdb_ids = rigid_docking.get_test_pdb_ids()
# rigid_docking.dock_molecule_pool(test_pdb_ids=test_pdb_ids)

# start_time = time.time()  
# rigid_docking.docking_analysis_pool()
# runtime = time.time() - start_time
# print(f'{runtime} seconds runtime')

print('All results')
rigid_docking.analysis_report(only_good_docking=False)

print('Only good docking results')
rigid_docking.analysis_report(only_good_docking=True)