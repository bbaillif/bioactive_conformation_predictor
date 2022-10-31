import torch
import numpy as np

from abc import abstractmethod, ABC
from rdkit.Chem.rdchem import Mol
from typing import Sequence, List, Any

class ConfRanker(ABC):
    
    def __init__(self,
                 name: str,
                 ascending: bool = True):
        self.name = name
        self.ascending = ascending
    
    def rank_molecule(self,
                      mol: Mol) -> Sequence[int]:
        input_list = self.get_input_list_from_mol(mol)
        ranks = self.rank_input_list(input_list)
        return ranks
    
    def rank_input_list(self,
                        input_list) -> Sequence[int]:
        values = self.get_values(input_list)
        ranks = self.get_ranks()
        return ranks
    
    def get_sorted_indexes(self,
                           values: Sequence[float]) -> Sequence[int]:
        if not self.ascending:
            values = -values # Reverse to sort in descending order
        argsort = np.argsort(values)
        return argsort
    
    def get_ranks(self,
                  values: Sequence[float]) -> Sequence[int]:
        argsort = self.get_sorted_indexes(values)
        ranks = np.argsort(argsort)
        return ranks
    
    def rank_confs(self,
                    mol: Mol) -> Mol:
        new_mol = Mol(mol)
        input_list = self.get_input_list_from_mol(mol)
        values = self.get_values(input_list)
        sorted_indexes = self.get_sorted_indexes(values)
        old_confs = [conf for conf in mol.GetConformers()]
        new_confs = []
        for i in sorted_indexes:
            new_confs.append(old_confs[i])
        new_mol.RemoveAllConformers()
        for conf in new_confs:
            new_mol.AddConformer(conf)
        return new_mol
        
    def select_conf_ratio(self,
                          mol,
                          ratio: int = 0.2,
                          ranked: bool = False) -> Mol:
        n_confs = mol.GetNumConformers()
        assert n_confs > 0, 'You need at least one conformer in the input mol'
        new_mol = Mol(mol)
        
        if not ranked:
            self.rank_confs(new_mol)
            
        min_i = int(n_confs * ratio) - 1
        confs = [conf for conf in mol.GetConformers()]
        for i in range(min_i, n_confs):
            conf = confs[i]
            conf_id = conf.GetId()
            new_mol.RemoveConformers(conf_id)
            
        return new_mol
                     
    
    def select_conf_number(self,
                           mol,
                           number: int = 10,
                           ranked: bool = False):
        n_confs = mol.GetNumConformers()
        assert n_confs > 0, 'You need at least one conf in the input mol'
        assert number < n_confs, 'The input number must be lower than the number of input confs in mol'
        new_mol = Mol(mol)
        
        if not ranked:
            self.rank_confs(new_mol)
            
        confs = [conf for conf in mol.GetConformers()]
        for i in range(number, n_confs):
            conf = confs[i]
            conf_id = conf.GetId()
            new_mol.RemoveConformers(conf_id)
            
        return new_mol
        
    
    @abstractmethod
    def get_input_list_from_mol(self,
                                mol: Mol) -> List[Any]:
        pass

    @abstractmethod
    def get_values(self,
                   input_list) -> Sequence[float]:
        pass
 
    # @abstractmethod
    # def get_input_list_from_data_list(self,
    #                                   data_list: Sequence[Any]) -> Sequence[Any]:
    #     pass
    
    # @abstractmethod
    # def get_output_list_from_data_list(self,
    #                                    data_list: Sequence[Any]) -> Sequence[Any]:
    #     pass
    
    