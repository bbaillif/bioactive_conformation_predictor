from typing import Dict, Any
from .atomistic import AtomicSchNet
from .atomistic_nn_model import AtomisticNNModel

class SchNetModel(AtomisticNNModel):
    """
    Class to setup an AtomisticNNModel using SchNet as backend
    
    :param config: Dictionnary of parameters. Must contain:
        num_interations: int = number of interation blocks
        cutoff: int = distance cutoff in Angstrom for neighbourhood convolutions
    :type config: Dict[str, Any]
    :param readout: Type of aggregation for atoms in a molecule. Default: sum
    :type readout:str
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 readout: str = 'add'):
        atomisctic_nn = AtomicSchNet(readout=readout, 
                                          num_interactions=config['num_interactions'],
                                          cutoff=config['cutoff'])
        AtomisticNNModel.__init__(self,
                                  config, 
                                  atomisctic_nn)
        
    @property
    def name(self) -> str:
        """
        :return: Name of the model
        :rtype: str
        """
        return 'SchNetModel'