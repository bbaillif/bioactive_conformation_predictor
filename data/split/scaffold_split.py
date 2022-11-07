import os
import random

from .molecule_split import MoleculeSplit
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import (MakeScaffoldGeneric, 
                                                 GetScaffoldForMol)
from collections import Counter
from typing import List

class ScaffoldSplit(MoleculeSplit):
    
    def __init__(self, 
                 split_type: str = 'scaffold', 
                 split_i: int = 0) -> None:
        super().__init__(split_type, split_i)
        
    def split_dataset(self,
                      frac_train = 0.8,
                      frac_val = 0.1):
        random.seed(42)
        all_smiles = self.cel_df['smiles'].unique()
        all_mols = [Chem.MolFromSmiles(smiles) for smiles in all_smiles]

        all_scaffolds = []
        correct_smiles = []
        for mol in all_mols :
            try :
                scaffold = self.get_scaffold(mol)
                all_scaffolds.append(scaffold)
                correct_smiles.append(Chem.MolToSmiles(mol))
            except Exception as e :
                print('Didnt work')
                print(str(e))

        counter = Counter(all_scaffolds)
        
        # Define the integer cutoff: minimal number of molecule in train
        # and in (train + val)
        train_cutoff = int(frac_train * len(correct_smiles))
        val_cutoff = int((frac_train + frac_val) * len(correct_smiles))

        unique_scaffolds = list(counter.keys())

        scaffold_splits_dir_path = os.path.join(self.splits_dir_path,
                                                self.split_type)
        if not os.path.exists(scaffold_splits_dir_path) :
            os.mkdir(scaffold_splits_dir_path)
        
        for i in range(5) :
            
            current_split_dir_path = os.path.join(scaffold_splits_dir_path, str(i))
            if not os.path.exists(current_split_dir_path):
                os.mkdir(current_split_dir_path)
    
            random.shuffle(unique_scaffolds)
            
            train_inds: List[int] = []
            val_inds: List[int] = []
            test_inds: List[int] = []
            
            # We first fill train, then val then test
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
            
    @staticmethod
    def get_scaffold(mol,
                     generic: bool = False) -> str:
        try :
            core = GetScaffoldForMol(mol)
            if generic :
                core = MakeScaffoldGeneric(mol=core)
            scaffold = Chem.MolToSmiles(core)
            return scaffold
        except Exception as e :
            print(str(e))
            raise Exception('Didnt work')