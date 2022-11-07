import os

from .molecule_split import MoleculeSplit
from sklearn.model_selection import train_test_split

class RandomSplit(MoleculeSplit):
    
    def __init__(self, 
                 split_type: str = 'random', 
                 split_i: int = 0) -> None:
        super().__init__(split_type, split_i)
        
    def split_dataset(self):
        all_smiles = self.cel_df['smiles'].unique()

        random_splits_dir_path = os.path.join(self.splits_dir_path, 
                                              self.split_type)
        if not os.path.exists(random_splits_dir_path) :
            os.mkdir(random_splits_dir_path)
        
        seed = 42
        for i in range(5) :
            
            current_split_dir_path = os.path.join(random_splits_dir_path, str(i))
            if not os.path.exists(current_split_dir_path):
                os.mkdir(current_split_dir_path)
                
            train_smiles, test_smiles = train_test_split(all_smiles, 
                                                        train_size=0.8, 
                                                        random_state=seed)
            val_smiles, test_smiles = train_test_split(test_smiles, 
                                                    train_size=0.5, 
                                                    random_state=seed)
            
            with open(os.path.join(current_split_dir_path, f'train_smiles.txt'), 'w') as f :
                for smiles in train_smiles :
                    f.write(smiles)
                    f.write('\n')

            with open(os.path.join(current_split_dir_path, f'val_smiles.txt'), 'w') as f :
                for smiles in val_smiles :
                    f.write(smiles)
                    f.write('\n')

            with open(os.path.join(current_split_dir_path, f'test_smiles.txt'), 'w') as f :
                for smiles in test_smiles :
                    f.write(smiles)
                    f.write('\n')
            
            seed = seed + 1