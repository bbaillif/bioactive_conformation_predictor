import numpy as np

from abc import abstractmethod, ABC
from rdkit.Chem import Mol
from typing import Sequence, List, Any

class ConfRanker(ABC):
    """Base class of conformer rankers

    :param name: Name of the ranker
    :type name: str
    :param ascending: Set to True to rank conformer by ascending
        values, or False for descending values, defaults to True
    :type ascending: bool, optional
    """
    
    def __init__(self,
                 name: str,
                 ascending: bool = True):
        
        assert name is not None, 'Ranker name must be defined'
        self.name = name
        self.ascending = ascending
        self.value_memory = {}
    
    
    def rank_molecule(self,
                      mol: Mol,
                      from_memory: bool = True
                      ) -> Sequence[int]:
        """Get ranks for conformers in the molecule

        :param mol: Input molecule
        :type mol: Mol
        :param from_memory: Set to True to get the values from memory,
            useful when value computation takes time, defaults to True
        :type from_memory: bool, optional
        :return: Ranks of each conformer
        :rtype: Sequence[int]
        """
        values = self.get_values_for_mol(mol, from_memory)
        ranks = self.get_ranks(values)
        return ranks
    
    
    def get_values_for_mol(self,
                            mol: Mol,
                            from_memory: bool=True
                            ) -> Sequence[float] :
        """Get values for conformers in the molecule

        :param mol: Input molecule
        :type mol: Mol
        :param from_memory: Set to True to get the values from memory,
            useful when value computation takes time, defaults to True
        :type from_memory: bool, optional
        :return: Values of each conformer
        :rtype: Sequence[float]
        """
        # Molecule name to store values if ranker is used multiple times
        mol_name = None
        if mol.HasProp('_Name'):
            mol_name = mol.GetProp('_Name')
            
        # Avoids building the input_list if mol name in memory
        values_in_memory = False
        if mol_name is not None and from_memory:
            if mol_name in self.value_memory:
                values = self.value_memory[mol_name]
                values_in_memory = True
            
        if not values_in_memory:
            input_list = self.get_input_list_for_mol(mol)
            values = self.compute_values(input_list)
            if mol_name is not None:
                self.value_memory[mol_name] = values
                
        return values
    
    def get_values_for_input_list(self,
                                    input_list: List[Any],
                                    mol_name: str = None
                                    ) -> Sequence[float]:
        """Get values for an input list of data. Each data represents one conf,
            i.e. a list of torch geometric data

        :param input_list: Input list of data
        :type input_list: List[Any]
        :param from_memory: Set to True to get the values from memory,
            useful when value computation takes time, defaults to True
        :type from_memory: bool, optional
        :return: Values of each conformer
        :rtype: Sequence[float]
        """
        
        # Avoids computing the values if mol name in memory
        values_in_memory = False
        if mol_name is not None:
            if mol_name in self.value_memory:
                values = self.value_memory[mol_name]
                values_in_memory = True
                
        if not values_in_memory:
            values = self.compute_values(input_list)
            if mol_name is not None:
                self.value_memory[mol_name] = values
            
        return values
    
    
    def rank_input_list(self,
                        input_list: List[Any],
                        mol_name: str = None
                        ) -> Sequence[int]:
        """Get ranks for an input list of data

        :param input_list: input list of data
        :type input_list: List[Any]
        :param mol_name: Name of the molecule, defaults to None
        :type mol_name: str, optional
        :return: Ranks for each conformer in the input list
        :rtype: Sequence[int]
        """
        values = self.get_values_for_input_list(input_list, 
                                                mol_name)
        ranks = self.get_ranks(values)
        return ranks
    
    
    def get_sorted_indexes(self,
                           values: Sequence[float]
                           ) -> Sequence[int]:
        """Get the sorted indexes (argsort) corresponding to sorted values (in
            ascending of descending values depending on self.ascending)

        :param values: Input values for each conformer
        :type values: Sequence[float]
        :return: List of indexes to sort the values
        :rtype: Sequence[int]
        """
        if not self.ascending:
            values = -values # Reverse to sort in descending order
        argsort = np.argsort(values)
        return argsort
    
    
    def get_ranks(self,
                  values: Sequence[float]
                  ) -> Sequence[int]:
        """Get the ranks based on input values

        :param values: Input values
        :type values: Sequence[float]
        :return: Ranks of each value (when sorted)
        :rtype: Sequence[int]
        """
        argsort = self.get_sorted_indexes(values)
        ranks = np.argsort(argsort)
        return ranks
    
    
    def rank_confs(self,
                    mol: Mol) -> Mol:
        """Sort conformers in the input molecule 

        :param mol: Input molecule
        :type mol: Mol
        :return: Molecule with conformers sorted
        :rtype: Mol
        """
        new_mol = Mol(mol)
        values = self.get_values_for_mol(mol)
        sorted_indexes = self.get_sorted_indexes(values)
        old_confs = [conf for conf in mol.GetConformers()]
        new_confs = []
        for i in sorted_indexes:
            new_confs.append(old_confs[i])
        new_mol.RemoveAllConformers()
        for conf in new_confs:
            new_mol.AddConformer(conf)
        return new_mol
        
        
    def select_conf_fraction(self,
                             mol: Mol,
                             fraction: float = 0.2,
                             ranked: bool = False) -> Mol:
        """Select a fraction of conformers from the input molecule

        :param mol: Input molecule
        :type mol: Mol
        :param fraction: Fraction of conformers to select, defaults to 0.2
        :type fraction: float, optional
        :param ranked: If True, ranks the conformers based on the values before 
            selecting the conformers, defaults to False
        :type ranked: bool, optional
        :return: Molecule with a fraction of conformers remaining
        :rtype: Mol
        """
        n_confs = mol.GetNumConformers()
        min_i = int(n_confs * fraction) - 1
        
        new_mol = self.select_conf_number(mol,
                                          number=min_i,
                                          ranked=ranked)

        return new_mol
                     
    
    def select_conf_number(self,
                           mol: Mol,
                           number: int = 10,
                           ranked: bool = False):
        """Select a number of conformers from the input molecule

        :param mol: Input molecule
        :type mol: Mol
        :param number: Number of conformers to select, defaults to 0.2
        :type number: float, optional
        :param ranked: If True, ranks the conformers based on the values before 
            selecting the conformers, defaults to False
        :type ranked: bool, optional
        :return: Molecule with a fraction of conformers remaining
        :rtype: Mol
        """
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
    def get_input_list_for_mol(self,
                                mol: Mol) -> List[Any]:
        pass


    @abstractmethod
    def compute_values(self,
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
    
    