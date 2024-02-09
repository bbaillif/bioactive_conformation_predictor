import os
import random

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds.MurckoScaffold import (MakeScaffoldGeneric, 
                                                 GetScaffoldForMol)
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
from .molecule_split import MoleculeSplit
from bioconfpred.data.utils import SimilaritySearch
from bioconfpred.params import (DATA_DIRPATH,
                    BIO_CONF_DIRNAME,
                    GEN_CONF_DIRNAME,
                    RMSD_DIRNAME,
                    SPLITS_DIRNAME)


random.seed(42)

class ScaffoldSplit(MoleculeSplit):
    """Class for the scaffold split: the ligands are clustered based on 
    Bemis-Murcko scaffolds (minimum intra-cluster Tanimoto similarity of ECFP6 
    is 50%), then the clustered are randomly shuffled between train, val and test
    subsets

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
                 split_type: str = 'scaffold', 
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
                      val_fraction: float = 0.1,
                      min_similarity: float = 0.5) -> None:
        """Performs the scaffold split of the ligands

        :param train_fraction: Fraction of train samples (between 0 and 1), 
            defaults to 0.8
        :type train_fraction: float, optional
        :param val_fraction: Fraction of validation samples (between 0 and 1), 
            defaults to 0.1
        :type val_fraction: float, optional
        :param min_similarity: Value of the mininum intra-cluster Tanimoto 
            similarity of ECFP6
        :type min_similarity: float
        """
        
        self.check_fractions(train_fraction, val_fraction)
        
        all_smiles = self.cel_df['smiles'].unique()

        scaffold_to_smiles = defaultdict(list)
        for smiles in all_smiles :
            mol = Chem.MolFromSmiles(smiles)
            scaffold = self.get_scaffold(mol)
            scaffold_to_smiles[scaffold].append(smiles)
        
        # Define the integer cutoff: minimal number of molecule in train
        # and in (train + val)
        train_cutoff = int(train_fraction * len(all_smiles))
        val_cutoff = int((train_fraction + val_fraction) * len(all_smiles))

        unique_scaffolds = list(scaffold_to_smiles.keys())
        
        ss = SimilaritySearch(unique_scaffolds)

        dm = 1 - ss.sim_matrix
        condensed_dm = squareform(dm)

        Z = linkage(condensed_dm)
        
        max_distance = 1 - min_similarity
        T = fcluster(Z, 
                     t=max_distance, 
                     criterion='distance')

        cluster_to_scaffolds = defaultdict(list)
        for scaffold, cluster_id in zip(unique_scaffolds, T):
            cluster_to_scaffolds[cluster_id].append(scaffold)
            
        unique_cluster_ids = list(cluster_to_scaffolds.keys())

        scaffold_splits_dir_path = os.path.join(self.splits_dir_path,
                                                self.split_type)
        if not os.path.exists(scaffold_splits_dir_path) :
            os.mkdir(scaffold_splits_dir_path)

        for i in range(5) :
            
            random.shuffle(unique_cluster_ids)
            
            current_split_dir_path = os.path.join(scaffold_splits_dir_path, str(i))
            if not os.path.exists(current_split_dir_path):
                os.mkdir(current_split_dir_path)
            
            train_smiles = []
            val_smiles = []
            test_smiles = []
            
            for current_cluster_id in unique_cluster_ids:
                scaffolds = cluster_to_scaffolds[current_cluster_id]
                cluster_smiles = []
                for scaffold in scaffolds:
                    smiles_list = scaffold_to_smiles[scaffold]
                    cluster_smiles.extend(smiles_list)
                if len(train_smiles) + len(cluster_smiles) > train_cutoff:
                    if len(train_smiles) + len(val_smiles) + len(cluster_smiles) > val_cutoff:
                        test_smiles.extend(cluster_smiles)
                    else:
                        val_smiles.extend(cluster_smiles)
                else:
                    train_smiles.extend(cluster_smiles)
            
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
    def get_scaffold(mol: Mol,
                     generic: bool = False) -> str:
        """Get scaffold (as a smiles) for molecule.

        :param mol: Input molecule
        :type mol: Mol
        :param generic: Set to True to obtain generic scaffold (all atoms are 
            carbon), defaults to False
        :type generic: bool, optional
        :raises Exception: If scaffold definition does not work
        :return: Smiles of the scaffold
        :rtype: str
        """
        try :
            core = GetScaffoldForMol(mol)
            if generic :
                core = MakeScaffoldGeneric(mol=core)
            scaffold = Chem.MolToSmiles(core)
            return scaffold
        except Exception as e :
            print(str(e))
            raise Exception('Didnt work')