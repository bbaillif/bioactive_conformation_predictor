import os
import pandas as pd

from sklearn.model_selection import train_test_split

root = '/home/bb596/hdd/pdbbind_bioactive/data/'
cel_name = 'pdb_conf_ensembles/'
cel_dir = os.path.join(root, cel_name)
cel_df_path = os.path.join(cel_dir, 'ensemble_names.csv')
cel_df = pd.read_csv(cel_df_path)

all_smiles = cel_df['smiles'].unique()

random_splits_dir_name = 'random_splits'
random_splits_dir_path = os.path.join(root, random_splits_dir_name)
if not os.path.exists(random_splits_dir_path) :
    os.mkdir(random_splits_dir_path)
    
seed = 42
for i in range(5) :
    train_smiles, test_smiles = train_test_split(all_smiles, train_size=0.8, random_state=seed)
    val_smiles, test_smiles = train_test_split(test_smiles, train_size=0.5, random_state=seed)
    
    with open(os.path.join(random_splits_dir_path, f'train_smiles_{i}.txt'), 'w') as f :
        for smiles in train_smiles :
            f.write(smiles)
            f.write('\n')

    with open(os.path.join(random_splits_dir_path, f'val_smiles_{i}.txt'), 'w') as f :
        for smiles in val_smiles :
            f.write(smiles)
            f.write('\n')

    with open(os.path.join(random_splits_dir_path, f'test_smiles_{i}.txt'), 'w') as f :
        for smiles in test_smiles :
            f.write(smiles)
            f.write('\n')
    
    seed = seed + 1