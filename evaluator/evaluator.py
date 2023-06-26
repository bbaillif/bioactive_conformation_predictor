import os
from abc import ABC, abstractmethod
from typing import Any, Sequence, Dict
from conf_ensemble import ConfEnsembleLibrary

class Evaluator(ABC):
    """Base class for evaluators that assess rankers/model on conformer
    ranking/RMSD prediction

    :param evaluation_name: Name of the current evaluation (split_name and split_i)
    :type evaluation_name: str
    :param results_dir: Directory where the results are stored
    :type results_dir: str
    """
    
    def __init__(self,
                 evaluation_name: str,
                 results_dir: str):
        
        self.evaluation_name = evaluation_name
        self.results_dir = results_dir
        self.evaluation_dir = os.path.join(self.results_dir, 
                                           self.evaluation_name)
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        
    @abstractmethod
    def evaluate_library(self,
                         cel: ConfEnsembleLibrary,
                         d_targets: Dict[str, Sequence[Any]]):
        pass