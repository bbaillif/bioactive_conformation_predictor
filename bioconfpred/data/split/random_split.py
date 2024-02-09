import os

from .molecule_split import MoleculeSplit
from sklearn.model_selection import train_test_split
from bioconfpred.params import (DATA_DIRPATH,
                    BIO_CONF_DIRNAME,
                    GEN_CONF_DIRNAME,
                    RMSD_DIRNAME,
                    SPLITS_DIRNAME)

class RandomSplit(MoleculeSplit):
    """Class for the random split: the ligands are randomly shuffled between the
    train, val and test subsets

    :param split_type: Name of the split type
    :type split_type: str
    :param split_i: Number of the split (iteration)
    :type split_i: int
    :param cel_name: Name of the bioactive conformation ensemble library, 
        defaults to 'pdb_conf_ensembles'
    :type cel_name: str, optional
    :param gen_cel_name: Name of the generated conformer ensemble library, 
        defaults to 'gen_conf_ensembles'
    :type gen_cel_name: str, optional
    :param root: Data directory
    :type root: str, optional
    :param splits_dirname: Name of the directory where the splits information
        are stored, defaults to 'splits'
    :type splits_dirname: str, optional
    :param rmsd_name: Name of the directory storing the precomputed RMSD, 
        defaults to 'rmsds'
    :type rmsd_name: str, optional
    """
    
    def __init__(self, 
                 split_type: str = 'random', 
                 split_i: int = 0,
                 cel_name: str = BIO_CONF_DIRNAME, 
                 gen_cel_name: str = GEN_CONF_DIRNAME,
                 root: str = DATA_DIRPATH, 
                 splits_dirname: str = SPLITS_DIRNAME, 
                 rmsd_name: str = RMSD_DIRNAME) -> None:

        super().__init__(split_type, 
                         split_i,
                         cel_name=cel_name,
                         gen_cel_name=gen_cel_name,
                         root=root,
                         splits_dirname=splits_dirname,
                         rmsd_name=rmsd_name)
        
    def split_dataset(self,
                      train_fraction: float = 0.8,
                      val_fraction: float = 0.1) -> None:
        """Performs the random split of the ligands

        :param train_fraction: Fraction of train samples (between 0 and 1), 
            defaults to 0.8
        :type train_fraction: float, optional
        :param val_fraction: Fraction of validation samples (between 0 and 1), 
            defaults to 0.1
        :type val_fraction: float, optional
        """
        
        self.check_fractions(train_fraction, val_fraction)
        
        print('Performing random split')
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
                                                        train_size=train_fraction, 
                                                        random_state=seed)
            new_train_size = val_fraction / (1 - train_fraction)
            val_smiles, test_smiles = train_test_split(test_smiles, 
                                                    train_size=new_train_size, 
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