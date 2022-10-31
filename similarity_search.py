import numpy as np
import time

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity

class SimilaritySearch() :
    
    def __init__(self, smiles_list) :
        self.smiles_list = list(set(smiles_list))
        self.mols = [Chem.MolFromSmiles(smiles) for smiles in self.smiles_list]
        self.fps = [self.get_morgan_fingerprint(mol) for mol in self.mols]
        self.sim_matrix = self.get_sim_matrix()
        print('Similarity matrix ready')
        
    def get_sim_matrix(self) :
#         sim_triangle = GetTanimotoSimMat(self.fps)
#         sim_matrix = self.tri2mat(sim_triangle)
#         return sim_matrix
        
        n_fps = len(self.fps)
        sim_matrix = np.eye(n_fps)
        for i, fp1 in enumerate(self.fps) :
            fp1 = self.fps[i]
            other_fps = self.fps[i+1:]
            sims = BulkTanimotoSimilarity(fp1, other_fps)
            for j, sim in enumerate(sims) :
                sim_matrix[i, i + 1 + j] = sim_matrix[i + 1 + j, i] = sim
        return sim_matrix
        
    def get_morgan_fingerprint(self, mol) :
        return AllChem.GetMorganFingerprintAsBitVect(mol, 3, useChirality=True)
        
    def get_similarity(self, fp1, fp2) :
        return TanimotoSimilarity(fp1, fp2)
        
    def find_closest_in_set(self, smiles, n=1) :
        
        #import pdb; pdb.set_trace()
        
        if smiles in self.smiles_list :
            smiles_index = self.smiles_list.index(smiles)
            sims = self.sim_matrix[smiles_index]
            sims = [sim for i, sim in enumerate(sims) if i != smiles_index]
            smiles_list = [s for i, s in enumerate(self.smiles_list) if i != smiles_index]
        else :
            input_mol = Chem.MolFromSmiles(smiles)
            input_fp = self.get_morgan_fingerprint(input_mol)
            sims = [self.get_similarity(input_fp, fp2) for fp2 in self.fps]
            smiles_list = self.smiles_list
            
        sims = np.array(sims)
        best_sim_indexes = np.argsort(-sims)[:n] # negate to get best
        closest_smiles = [smiles_list[i] for i in best_sim_indexes]
        closest_sims = sims[best_sim_indexes]
        return closest_smiles, closest_sims
    
    
    def find_closest_in_subset(self,
                               smiles: str,
                               subset: list,
                               n: int = 1) :
        assert all([s in self.smiles_list for s in subset]), \
            'One or multiple smiles from subset are not in the database'
        start_time = time.time()
        subset_idxs = []
        smiles_list = []
        for i, s in enumerate(self.smiles_list):
            if s != smiles:
                subset_idxs.append(i)
                smiles_list.append(s)
        
        # subset_sim_matrix = self.sim_matrix[np.ix_(subset_idxs,subset_idxs)]
    
        if smiles in self.smiles_list :
            smiles_index = self.smiles_list.index(smiles)
            sims = self.sim_matrix[smiles_index][subset_idxs]
        else :
            input_mol = Chem.MolFromSmiles(smiles)
            input_fp = self.get_morgan_fingerprint(input_mol)
            fps = [fp 
                   for i, fp 
                   in self.fps
                   if i in subset_idxs]
            sims = [self.get_similarity(input_fp, fp2) for fp2 in fps]
            
        sims = np.array(sims)
        best_sim_indexes = np.argsort(-sims)[:n] # negate to get best
        closest_smiles = [smiles_list[i] for i in best_sim_indexes]
        closest_sims = sims[best_sim_indexes]
        return closest_smiles, closest_sims
        
    def tri2mat(self, tri_arr):
        n = len(tri_arr)
        m = int((np.sqrt(1 + 4 * 2 * n) + 1) / 2)
        arr = np.ones([m, m])
        for i in range(m):
            for j in range(i):
                arr[i][j] = tri_arr[i + j - 1]
                arr[j][i] = tri_arr[i + j - 1]
        return arr