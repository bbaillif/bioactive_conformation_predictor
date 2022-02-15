import os
from multiprocessing import Pool


def dock_smiles(params) :
    i, smiles = params
    print('OK')

data_dir = 'data/'
with open(os.path.join(data_dir, 'random_splits', f'test_smiles_random_split_0.txt'), 'r') as f :
    test_smiles = f.readlines()
    test_smiles = [smiles.strip() for smiles in test_smiles]
    
for i, smiles in enumerate(test_smiles[:100]) :
    
    with Pool(10) as pool :
        pool.map(dock_smiles, [(i, smiles)])