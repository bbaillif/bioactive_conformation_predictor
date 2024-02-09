from typing import Dict, Any
from .atomistic import AtomicDimeNet
from .atomistic_nn_model import AtomisticNNModel
from bioconfpred.data.split import DataSplit
from bioconfpred.params import (LOG_DIRPATH, 
                    DIMENET_CONFIG,
                    DIMENET_MODEL_NAME)

class DimeNetModel(AtomisticNNModel):
    """
    Class to setup an AtomisticNNModel using DimeNet as backend
    
    :param config: Dictionnary of parameters. Must contain:
        see list of arguments
    :type config: Dict[str, Any]
    :param readout: Type of aggregation for atoms in a molecule. Default: sum
    :type readout:str
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 readout: str = 'add'):
        atomisctic_nn = AtomicDimeNet(readout=readout, 
                                      hidden_channels=config['hidden_channels'],
                                      out_channels=config['out_channels'],
                                      num_blocks=config['num_blocks'],
                                      int_emb_size=config['int_emb_size'],
                                      basis_emb_size=config['basis_emb_size'],
                                      out_emb_channels=config['out_emb_channels'],
                                      num_spherical=config['num_spherical'],
                                      num_radial=config['num_radial'])
        AtomisticNNModel.__init__(self,
                                  config, 
                                  atomisctic_nn)
        
    
    @property
    def name(self) -> str:
        """
        :return: Name of the model
        :rtype: str
        """
        return DIMENET_MODEL_NAME
    
    
    @classmethod
    def get_model_for_data_split(cls,
                                 data_split: DataSplit,
                                 log_dir: str = LOG_DIRPATH
                                 ) -> 'DimeNetModel':
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
        config = DIMENET_CONFIG
        config['data_split'] = data_split
        return cls._get_model_for_data_split(data_split=data_split,
                                            model_name=DIMENET_MODEL_NAME,
                                            config=config, 
                                            log_dir=log_dir)