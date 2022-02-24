from rigid_docking import RigidDocking
import os
import time
import pickle

with open('data/raw/ccdc_generated_conf_ensemble_library.p', 'rb') as f:
            conf_ensemble_library = pickle.load(f)

rigid_docking = RigidDocking()

# pdbbind_docking.prepare_analysis()
# pdb_id = '6etj'
# rdkit_native_ligand = pdbbind_docking.get_native_ligand(pdb_id, 
#                                                   conf_ensemble_library)
# pdbbind_docking.analyze_pdb_id(pdb_id=pdb_id,
#                                rdkit_native_ligand=rdkit_native_ligand)

start_time = time.time()  
rigid_docking.docking_analysis_pool()
runtime = time.time() - start_time
print(f'{runtime} seconds runtime')

rigid_docking.analysis_report()