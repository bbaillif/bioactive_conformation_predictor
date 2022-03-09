from rigid_docking import RigidDocking
import os
import time
import pickle

rigid_docking = RigidDocking(dataset='platinum')

# with open('data/raw/ccdc_generated_conf_ensemble_library.p', 'rb') as f:
#     conf_ensemble_library = pickle.load(f)
# rigid_docking.prepare_analysis()
# pdb_id = '6etj'
# rdkit_native_ligand = rigid_docking.get_native_ligand(pdb_id, 
#                                                   conf_ensemble_library)
# rigid_docking.analyze_pdb_id(pdb_id=pdb_id,
#                                rdkit_native_ligand=rdkit_native_ligand)

# start_time = time.time()  
# rigid_docking.docking_analysis_pool()
# runtime = time.time() - start_time
# print(f'{runtime} seconds runtime')

rigid_docking.analysis_report()