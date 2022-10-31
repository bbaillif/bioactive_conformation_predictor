import os
import random
import pandas as pd

from rdkit import Chem
from typing import List
from collections import Counter
from rdkit.Chem.Scaffolds.MurckoScaffold import (MakeScaffoldGeneric, 
                                                 GetScaffoldForMol)

random.seed(42)
root = '/home/bb596/hdd/pdbbind_bioactive/data/'
cel_name = 'pdb_conf_ensembles/'
cel_dir = os.path.join(root, cel_name)
cel_df_path = os.path.join(cel_dir, 'ensemble_names.csv')
cel_df = pd.read_csv(cel_df_path)

all_smiles = cel_df['smiles'].unique()
all_mols = [Chem.MolFromSmiles(smiles) for smiles in all_smiles]

def get_scaffold(mol, 
                 generic: bool = False) -> str :
    try :
        core = GetScaffoldForMol(mol)
        if generic :
            core = MakeScaffoldGeneric(mol=core)
        scaffold = Chem.MolToSmiles(core)
        return scaffold
    except Exception as e :
        print(str(e))
        raise Exception('Didnt work')
    
all_scaffolds = []
correct_smiles = []
for mol in all_mols :
    try :
        scaffold = get_scaffold(mol)
        all_scaffolds.append(scaffold)
        correct_smiles.append(Chem.MolToSmiles(mol))
    except Exception as e :
        print('Didnt work')
        print(str(e))
        
counter = Counter(all_scaffolds)

frac_train = 0.8
frac_val = 0.1

train_cutoff = int(frac_train * len(correct_smiles))
val_cutoff = int((frac_train + frac_val) * len(correct_smiles))
train_inds = []
val_inds = []
test_inds = []

unique_scaffolds = list(counter.keys())

scaffold_splits_dir_name = 'scaffold_splits'
scaffold_splits_dir_path = os.path.join(root, scaffold_splits_dir_name)
if not os.path.exists(scaffold_splits_dir_path) :
    os.mkdir(scaffold_splits_dir_path)
    
for i in range(5) :
    
    random.shuffle(unique_scaffolds)
    
    train_inds: List[int] = []
    val_inds: List[int] = []
    test_inds: List[int] = []
    
    for scaffold in unique_scaffolds:
        indices = [i for i, s in enumerate(all_scaffolds) if s == scaffold]
        if len(train_inds) + len(indices) > train_cutoff:
            if len(train_inds) + len(val_inds) + len(indices) > val_cutoff:
                test_inds += indices
            else:
                val_inds += indices
        else:
            train_inds += indices
            
    train_smiles = [smiles 
                    for i, smiles in enumerate(correct_smiles) 
                    if i in train_inds]
    val_smiles = [smiles 
                  for i, smiles in enumerate(correct_smiles) 
                  if i in val_inds]
    test_smiles = [smiles 
                   for i, smiles in enumerate(correct_smiles) 
                   if i in test_inds]
    
    with open(os.path.join(scaffold_splits_dir_path, f'train_smiles_{i}.txt'), 'w') as f :
        for smiles in train_smiles :
            f.write(smiles)
            f.write('\n')
        
    with open(os.path.join(scaffold_splits_dir_path, f'val_smiles_{i}.txt'), 'w') as f :
        for smiles in val_smiles :
            f.write(smiles)
            f.write('\n')
        
    with open(os.path.join(scaffold_splits_dir_path, f'test_smiles_{i}.txt'), 'w') as f :
        for smiles in test_smiles :
            f.write(smiles)
            f.write('\n')