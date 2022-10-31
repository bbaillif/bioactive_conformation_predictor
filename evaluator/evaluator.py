import os
from abc import ABC, abstractmethod
from typing import List, Any, Sequence, Dict
from conf_ensemble.conf_ensemble_library import ConfEnsembleLibrary

class Evaluator(ABC):
    
    def __init__(self,
                 evaluation_name,
                 results_dir):
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