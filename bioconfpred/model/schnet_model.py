from typing import Dict, Any
from .atomistic import AtomicSchNet
from .atomistic_nn_model import AtomisticNNModel
from bioconfpred.data.split import DataSplit
from bioconfpred.params import (LOG_DIRPATH, 
                    SCHNET_CONFIG,
                    SCHNET_MODEL_NAME)

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
        self.name = SCHNET_MODEL_NAME
        
    @property
    def name(self) -> str:
        """
        :return: Name of the model
        :rtype: str
        """
        return self._name
    
    
    @name.setter
    def name(self, name):
        self._name = name
    
    
    @classmethod
    def get_model_for_data_split(cls,
                                 data_split: DataSplit,
                                 log_dir: str = LOG_DIRPATH
                                 ) -> 'SchNetModel':
        """Get the trained model for a given data split

        :param data_split: Data split
        :type data_split: DataSplit
        :param root: Data directory
        :type root: str
        :param log_dir: Directory where training log are stored
        :type log_dir: str, optional
        :return: Trained model
        :rtype: SchNetModel
        """
        config = SCHNET_CONFIG
        config['data_split'] = data_split
        return cls._get_model_for_data_split(data_split=data_split,
                                            model_name=SCHNET_MODEL_NAME,
                                            config=config, 
                                            log_dir=log_dir)