import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from conf_ensemble import ConfEnsemble

descriptor_names = ['E', 'E_sol', 'rgyr', 'ASA_H', 'ASA_P']
moe_path = '/usr/local/ucc/moe/moe2022/'
n_tokens = 2

root = '/home/bb596/hdd/pdbbind_bioactive/data'
pdb_conf_ensembles_dirpath = os.path.join(root, 'pdb_conf_ensembles/')
gen_conf_ensembles_dirpath = os.path.join(root, 'gen_conf_ensembles/')

pdb_conf_ensembles_moe_dirpath = os.path.join(root, 'pdb_conf_ensembles_moe/')
if not os.path.exists(pdb_conf_ensembles_moe_dirpath):
    os.mkdir(pdb_conf_ensembles_moe_dirpath)

gen_conf_ensembles_moe_dirpath = os.path.join(root, 'gen_conf_ensembles_moe/')
if not os.path.exists(gen_conf_ensembles_moe_dirpath):
    os.mkdir(gen_conf_ensembles_moe_dirpath)

def compute_moe_ensemble(input_filepath, output_filepath):
    descriptor_str = ','.join(descriptor_names)
    ff_path = os.path.join(moe_path, 'lib', 'mmff94x.ff.gz')
    command = f'{moe_path}/bin/sddesc -calc {descriptor_str} {input_filepath} -o {output_filepath} -mpu {n_tokens} -forcefield {ff_path}'
    # import pdb;pdb.set_trace()
    os.system(command)
    
cel_df_path = os.path.join(pdb_conf_ensembles_dirpath, 'ensemble_names.csv')
cel_df = pd.read_csv(cel_df_path)

# for i, row in tqdm(cel_df.iterrows(), total=cel_df.shape[0]):
#     name = row['ensemble_name']
#     filename = row['filename']
    
#     output_filepath = os.path.join(pdb_conf_ensembles_moe_dirpath, filename)
#     input_filepath = os.path.join(pdb_conf_ensembles_dirpath, filename)
#     compute_moe_ensemble(input_filepath, output_filepath)
    
#     output_filepath = os.path.join(gen_conf_ensembles_moe_dirpath, filename)
#     input_filepath = os.path.join(gen_conf_ensembles_dirpath, filename)
#     compute_moe_ensemble(input_filepath, output_filepath)


def compute_delta_energies(mol,
                           e_hyd_factor: float = 0.1,
                           e_hphi_factor: float = 0.1,
                           e_rgyr_factor: float = 25.0):
    
    energy = []
    e_sol = []
    asa_h = []
    asa_p = []
    rgyr = []
    for conf in mol.GetConformers():
        energy.append(conf.GetDoubleProp('E'))
        rgyr.append(conf.GetDoubleProp('rgyr'))
        e_sol.append(conf.GetDoubleProp('E_sol'))
        asa_h.append(conf.GetDoubleProp('ASA_H'))
        asa_p.append(conf.GetDoubleProp('ASA_P'))
    energy = np.array(energy)
    rgyr = np.array(rgyr)
    e_sol = np.array(e_sol)
    asa_h = np.array(asa_h)
    asa_p = np.array(asa_p)
    
    min_energy = energy.min()
    delta_u = energy - min_energy
    original_delta_e_sol = delta_u + e_sol # as mentionned in Habgood 2017
    
    min_e_sol = e_sol.min()
    e_sol_diff = e_sol - min_e_sol
    delta_e_sol = delta_u + e_sol_diff
    
    e_hyd = asa_h * e_hyd_factor
    delta_e_hyd = delta_u + e_hyd
    
    e_hphi = asa_p * e_hphi_factor
    delta_e_hphi = delta_u + e_hphi
    
    e_rgyr = rgyr * e_rgyr_factor
    delta_e_rgyr = delta_u + e_rgyr
    
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetDoubleProp('delta_u', delta_u[i])
        conf.SetDoubleProp('delta_e_sol', delta_e_sol[i])
        conf.SetDoubleProp('original_delta_e_sol', original_delta_e_sol[i])
        conf.SetDoubleProp('delta_e_hyd', delta_e_hyd[i])
        conf.SetDoubleProp('delta_e_hphi', delta_e_hphi[i])
        conf.SetDoubleProp('delta_e_rgyr', delta_e_rgyr[i])

pdb_conf_ensembles_moe_all_dirpath = os.path.join(root, 'pdb_conf_ensembles_moe_all/')
if not os.path.exists(pdb_conf_ensembles_moe_all_dirpath):
    os.mkdir(pdb_conf_ensembles_moe_all_dirpath)

gen_conf_ensembles_moe_all_dirpath = os.path.join(root, 'gen_conf_ensembles_moe_all/')
if not os.path.exists(gen_conf_ensembles_moe_all_dirpath):
    os.mkdir(gen_conf_ensembles_moe_all_dirpath)

for i, row in tqdm(cel_df.iterrows(), total=cel_df.shape[0]):
    name = row['ensemble_name']
    filename = row['filename']
    
    try:
        pdb_ce_filepath = os.path.join(pdb_conf_ensembles_moe_dirpath, filename)
        ce = ConfEnsemble.from_file(filepath=pdb_ce_filepath, 
                                    name=name)
        compute_delta_energies(ce.mol) # Add energies to each conf
    except:
        print(f'Delta energy computation failed for {name}')
    else:
        filepath = os.path.join(pdb_conf_ensembles_moe_all_dirpath, filename)
        ce.save_ensemble(sd_writer_path=filepath)
    
    try:
        gen_ce_filepath = os.path.join(gen_conf_ensembles_moe_dirpath, filename)
        ce = ConfEnsemble.from_file(filepath=gen_ce_filepath, 
                                    name=name)
        compute_delta_energies(ce.mol) # Add energies to each conf
    except:
        print(f'Delta energy computation failed for {name}')
    else:
        filepath = os.path.join(gen_conf_ensembles_moe_all_dirpath, filename)
        ce.save_ensemble(sd_writer_path=filepath)
    
cel_df.to_csv(os.path.join(gen_conf_ensembles_moe_all_dirpath, 'ensemble_names.csv'))
cel_df.to_csv(os.path.join(pdb_conf_ensembles_moe_all_dirpath, 'ensemble_names.csv'))
cel_df.to_csv(os.path.join(gen_conf_ensembles_moe_dirpath, 'ensemble_names.csv'))
cel_df.to_csv(os.path.join(pdb_conf_ensembles_moe_dirpath, 'ensemble_names.csv'))
    
# os.system('$MOE/bin/sddesc -calc vol,VSA,vsurf_A,vsurf_CP gen_conf_ensembles/0.sdf -o moe_data/0.sdf')