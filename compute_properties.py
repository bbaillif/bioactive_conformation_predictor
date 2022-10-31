import os
from types import new_class
import numpy as np
import pandas as pd

from tqdm import tqdm
from conf_ensemble import ConfEnsemble
from rdkit.Chem.rdMolDescriptors import (CalcNumRotatableBonds, 
                                         GetMorganFingerprint)
from rdkit.DataStructs import BulkTanimotoSimilarity
from data import MoleculeSplit, ProteinSplit

root = '../hdd/pdbbind_bioactive/data/'
cel_name = 'pdb_conf_ensembles'
cel_df_path = os.path.join(root, cel_name, 'ensemble_names.csv')
cel_df = pd.read_csv(cel_df_path)

names = []
n_rot_bonds = []
n_heavy_atoms = []
n_bioactives = []
n_generateds = []
ecfps = []
for i, row in tqdm(cel_df.iterrows(), total=cel_df.shape[0]):
    name = row['ensemble_name']
    filename = row['filename']
    try:
        filepath = os.path.join(root, 'gen_conf_ensembles', filename)
        ce = ConfEnsemble.from_file(filepath)
        n_heavy_atom = ce.mol.GetNumHeavyAtoms()
        n_rot_bond = CalcNumRotatableBonds(ce.mol)
        n_generated = ce.mol.GetNumConformers()
        ecfp = GetMorganFingerprint(ce.mol, 3, useChirality=True)
        
        filepath = os.path.join(root, 'pdb_conf_ensembles', filename)
        ce = ConfEnsemble.from_file(filepath)
        n_bioactive = ce.mol.GetNumConformers()
    except:
        print(f'{name} failed')
    else:
        names.append(name)
        n_heavy_atoms.append(n_heavy_atom)
        n_rot_bonds.append(n_rot_bond)
        n_bioactives.append(n_bioactive)
        n_generateds.append(n_generated)
        ecfps.append(ecfp)

property_dict = {
    'ensemble_name' : names,
    'number_heavy_atoms': n_heavy_atoms,
    'number_rotatable_bonds': n_rot_bonds,
    'number_bioactive_confs': n_bioactives,
    'number_generated_confs': n_generateds
}
property_df = pd.DataFrame(property_dict)
property_df = property_df.set_index('ensemble_name')

print('Computing Tanimoto similarity matrix')
n = len(ecfps)
ecfp_sim_matrix = np.ones((n, n))
for i, ecfp in tqdm(enumerate(ecfps)):
    sims = BulkTanimotoSimilarity(ecfp, ecfps[i+1:])
    ecfp_sim_matrix[i, i+1:] = ecfp_sim_matrix[i+1:, i] = sims
    
ecfp_sim_df = pd.DataFrame(ecfp_sim_matrix, index=names, columns=names)
ecfp_sim_df_path = os.path.join(root, 'ecfp_sim_df.csv')
ecfp_sim_df.to_csv(ecfp_sim_df_path)

    
for split_type in ['random', 'scaffold', 'protein']:
    
    for split_i in range(5):
        
        if split_type == 'protein':
            data_split = ProteinSplit(split_i=split_i)
        else:
            data_split = MoleculeSplit(split_type, split_i)
            
        train_smiles = data_split.get_smiles('train')
        train_names = cel_df[cel_df['smiles'].isin(train_smiles)]['ensemble_name'].values
        
        test_smiles = data_split.get_smiles('test')
        test_names = cel_df[cel_df['smiles'].isin(test_smiles)]['ensemble_name'].values
        
        train_name_i = [i for i, name in enumerate(names) if name in train_names]
        test_name_i = [i for i, name in enumerate(names) if name in test_names]
        
        d = {}
        for name in test_names:
            if name in names:
                i = names.index(name)
                max_sim = ecfp_sim_matrix[i][train_name_i].max()
                d[name] = max_sim
        
        # subset_matrix = ecfp_sim_matrix[test_name_i][train_name_i]
        # max_sims = subset_matrix.max(1)
        # d = {}
        # for name, sim in zip(test_names, max_sims):
        #     d[name] = sim
        
        series_name = f'{split_type}_split_{split_i}'
        property_df = pd.concat([property_df, pd.Series(d, name=series_name)], axis=1)
        
property_df_path = os.path.join(root, 'property_df.csv')
property_df.to_csv(property_df_path)
        