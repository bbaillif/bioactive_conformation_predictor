import torch
from abc import ABC, abstractmethod

class AtomisticNN(ABC):
    """
    Base class for atomistic neural network: taking as input a conformation
    and returning values for each atom of the molecule
    
    :param readout: Readout function to perform on the list of individual
        atomic values
    :type readout: str
    """
    
    def __init__(self,
                 readout: str = 'add') -> None:
        self.readout = readout
        
    
    @abstractmethod
    def forward(self) -> torch.Tensor:
        pass