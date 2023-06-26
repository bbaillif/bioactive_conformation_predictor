import numpy as np
import time

from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import (GetMorganFingerprintAsBitVect)
from rdkit.DataStructs.cDataStructs import (TanimotoSimilarity,
                                            BulkTanimotoSimilarity)

class SimilaritySearch() :
    """Allow to search similar molecules in a setup list

    :param smiles_list: List of smiles to store
    :type smiles_list: List[str]
    """
    
    def __init__(self, 
                 smiles_list: List[str]) -> None :
        
        self.smiles_list = list(set(smiles_list))
        self.mols = [Chem.MolFromSmiles(smiles) for smiles in self.smiles_list]
        self.fps = [GetMorganFingerprintAsBitVect(mol, radius=3, useChirality=True) 
                    for mol in self.mols]
        self.sim_matrix = self.get_sim_matrix()
        print('Similarity matrix ready')
        
    def get_sim_matrix(self) -> np.ndarray :
        """Returns the similarity matrix of the molecules from the smiles list

        :return: Square similarity matrix
        :rtype: np.ndarray
        """
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
        
        
    def find_closest_in_set(self, 
                            smiles: str, 
                            n: int = 1,
                            allow_itself: bool = False,
                            ) -> Tuple[List[Mol], List[float]]:
        """Find the n closest molecule(s) in the smiles list for the input smiles

        :param smiles: Input SMILES
        :type smiles: str
        :param n: Number of closest molecules to return, defaults to 1
        :type n: int, optional
        :param allow_itself: If SMILES is already in the list, returns itself if True
        and return the closest among others if False, defaults to False
        :type allow_itself: bool, optional
        :return: List of closest smiles, and corresponding similarities
        :rtype: Tuple[List[Mol], List[float]]
        """
        
        #import pdb; pdb.set_trace()
        
        if smiles in self.smiles_list :
            smiles_index = self.smiles_list.index(smiles)
            sims = self.sim_matrix[smiles_index]
            if allow_itself:
                smiles_index = -1 # index is always positive, therefore it wont be removed in next line
            sims = [sim 
                    for i, sim in enumerate(sims) 
                    if i != smiles_index]
            smiles_list = [s 
                           for i, s in enumerate(self.smiles_list) 
                           if i != smiles_index]
        else :
            input_mol = Chem.MolFromSmiles(smiles)
            input_fp = GetMorganFingerprintAsBitVect(mol=input_mol, 
                                                    radius=3, 
                                                    useChirality=True)
            sims = [TanimotoSimilarity(input_fp, fp2) 
                    for fp2 in self.fps]
            smiles_list = self.smiles_list
            
        sims = np.array(sims)
        best_sim_indexes = np.argsort(-sims)[:n] # negate to get best
        closest_smiles = [smiles_list[i] 
                          for i in best_sim_indexes]
        closest_sims = sims[best_sim_indexes]
        return closest_smiles, closest_sims
    
    
    def find_closest_in_subset(self,
                               smiles: str,
                               subset: List[str],
                               n: int = 1,
                               allow_itself: bool = False
                               ) -> Tuple[List[Mol], List[float]]:
        """Find the n closest molecule(s) in a given subset of the smiles list 
        for the input smiles

        :param smiles: Input SMILES
        :type smiles: str
        :param subset: List of SMILES, subset of the full list
        :type subset: List[str]
        :param n: Number of closest molecules to return, defaults to 1
        :type n: int, optional
        :param allow_itself: If SMILES is already in the list, returns itself if True
        and return the closest among others if False, defaults to False
        :type allow_itself: bool, optional
        :return: List of closest smiles, and corresponding similarities
        :rtype: Tuple[List[Mol], List[float]]
        """
        
        assert all([s in self.smiles_list for s in subset]), \
            'One or multiple smiles from subset are not in the database'
        start_time = time.time()
        subset_idxs = []
        smiles_list = []
        for i, s in enumerate(self.smiles_list):
            if s in subset:
                if s != smiles or allow_itself:
                    subset_idxs.append(i)
                    smiles_list.append(s)
        
        # subset_sim_matrix = self.sim_matrix[np.ix_(subset_idxs,subset_idxs)]
    
        if smiles in self.smiles_list :
            smiles_index = self.smiles_list.index(smiles)
            sims = self.sim_matrix[smiles_index][subset_idxs]
        else :
            input_mol = Chem.MolFromSmiles(smiles)
            input_fp = GetMorganFingerprintAsBitVect(mol=input_mol, 
                                                    radius=3, 
                                                    useChirality=True)
            fps = [fp 
                   for i, fp 
                   in enumerate(self.fps)
                   if i in subset_idxs]
            sims = [self.TanimotoSimilarity(input_fp, fp2) 
                    for fp2 in fps]
            
        sims = np.array(sims)
        best_sim_indexes = np.argsort(-sims)[:n] # negate to get best
        closest_smiles = [smiles_list[i] 
                          for i in best_sim_indexes]
        closest_sims = sims[best_sim_indexes]
        return closest_smiles, closest_sims
        