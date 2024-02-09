from bioconfpred.data.split import DataSplit
from typing import Dict, Any
from .atomistic.comenet import AtomicComENet
from .atomistic_nn_model import AtomisticNNModel
from bioconfpred.params import (LOG_DIRPATH, 
                    COMENET_CONFIG,
                    COMENET_MODEL_NAME)

class ComENetModel(AtomisticNNModel):
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
        atomisctic_nn = AtomicComENet(readout=readout)
        AtomisticNNModel.__init__(self,
                                  config, 
                                  atomisctic_nn)
        
    @property
    def name(self) -> str:
        """
        :return: Name of the model
        :rtype: str
        """
        return COMENET_MODEL_NAME
    
    
    @classmethod
    def get_model_for_data_split(cls,
                                 data_split: DataSplit,
                                 log_dir: str = LOG_DIRPATH
                                 ) -> 'ComENetModel':
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
        config = COMENET_CONFIG
        config['data_split'] = data_split
        return cls._get_model_for_data_split(data_split=data_split,
                                            model_name=COMENET_MODEL_NAME,
                                            config=config, 
                                            log_dir=log_dir)