import numpy as np
import torch

from abc import abstractmethod
from torch_geometric.data import Batch
from ccdc_rdkit_connector import CcdcRdkitConnector

class PoseSelector():

    def __init__(self,
                 selector_name: str,
                 ratio: int=0.2,
                 number: int=None):
        
        self.selector_name = selector_name
        self.ratio = ratio
        self.number = number

    @abstractmethod
    def select_poses(self,
                     poses):
        pass
    
    def filter_subset(self,
                      poses,
                      values,
                      ascending=True):
        values = np.array(values)
        
        if not ascending :
            values = -values
            
        sorted_indexes = values.argsort()
            
        if self.number is not None:
            assert self.number < len(values), \
            'Requested number must be lower than number of values'
            limit = self.number
        else :
            limit = int(len(values) * self.ratio)
        selected_indexes = sorted_indexes[:limit]
        poses_subset = [pose 
                        for i, pose in enumerate(poses) 
                        if i in selected_indexes]
        
        return poses_subset
        
        
class RandomPoseSelector(PoseSelector) :
    
    def __init__(self,
                 selector_name: str='random',
                 ratio: int=0.2,
                 number=None):
        super().__init__(selector_name, ratio, number)
        
    def select_poses(self,
                     poses):
        n_poses = len(poses)
        random_values = np.random((n_poses))
        poses_subset = self.filter_subset(poses, values=random_values)
            
        return poses_subset
  
class ScorePoseSelector(PoseSelector):
    
    def __init__(self, 
                 selector_name: str='score',
                 ratio: int=0.2,
                 number=None):
        super().__init__(selector_name, ratio, number)
    
    def select_poses(self,
                     poses,):
        scores = [pose.fitness for pose in poses]
        poses_subset = self.filter_subset(poses, 
                                          values=scores, 
                                          ascending=False)
            
        return poses_subset
        
class EnergyPoseSelector(PoseSelector):
    
    def __init__(self, 
                 mol_featurizer,
                 selector_name: str='energy',
                 ratio: int=0.2,
                 number=None):
        super().__init__(selector_name, ratio, number)
        self.mol_featurizer = mol_featurizer
        
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
    
    def select_poses(self,
                     poses,):
        rdkit_mol = self.ccdc_rdkit_connector.ccdc_ensemble_to_rdkit_mol(
            ccdc_ensemble=poses
            )

        data_list = self.mol_featurizer.featurize_mol(rdkit_mol)
        energies = [data.energy for data in data_list]
        poses_subset = self.filter_subset(poses, values=energies)
            
        return poses_subset
        
class ModelPoseSelector(PoseSelector):
    
    def __init__(self, 
                model,
                mol_featurizer,
                selector_name: str='model',
                ratio: int=0.2,
                number=None):
        super().__init__(selector_name, ratio, number)
        self.model = model
        self.mol_featurizer = mol_featurizer
        
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
    
    def select_poses(self,
                     poses,):
        rdkit_mol = self.ccdc_rdkit_connector.ccdc_ensemble_to_rdkit_mol(
            ccdc_ensemble=poses
            )

        data_list = self.mol_featurizer.featurize_mol(rdkit_mol)
        batch = Batch.from_data_list(data_list)

        with torch.no_grad() :
            preds = self.model(batch).cpu().numpy()
            preds = preds.reshape(-1)
        
        top20_preds = preds.argsort()[:20]
        poses_subset = [pose for i, pose in enumerate(poses) if i in top20_preds]
            
        return poses_subset