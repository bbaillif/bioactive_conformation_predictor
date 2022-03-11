from xdrlib import ConversionError
import numpy as np
import torch
import copy

from abc import abstractmethod
from torch_geometric.data import Batch
from ccdc_rdkit_connector import CcdcRdkitConnector, ConversionError


class PoseSelectionError(Exception):
    pass

class Pose() :
    """Class to store poses, simplifying the DockedLigand object from ccdc
    to only a molecule and a score
    
    :param pose: entry corresponding to docking pose
    :type pose: ccdc.io.Entry or rdkit.Chem.rdchem.Mol
    """
    
    def __init__(self,
                 pose,
                 backend='rdkit') -> None:
        self.backend = backend
        if backend == 'ccdc' :
            self.molecule = pose.molecule
            self.score = float(pose.attributes['Gold.PLP.Fitness'])
            self.identifier = pose.identifier
        elif backend == 'rdkit' :
            self.molecule = pose
            self.score = float(pose.GetProp('Gold.PLP.Fitness').strip())
            self.identifier = pose.GetProp('_Name')
            
    def fitness(self):
        return self.score

class PoseSelector():
    """Base class for pose selectors, that can sort poses depending on values,
    and select topN poses for best values
    
    :param selector_name: Name given to the selector, to differentiate different
        selectors in downstream tasks
    :type selector_name: str
    :param ratio: Ratio of poses that are selected by select_poses()
    :type ratio: float
    :param number: Number of poses that are selected by select_poses(). Default
        is None to use the ratio, uses the number if defined
    :type number: int
    """

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
        """Abstract class, to be implemented in inherited classes"""
        pass
    
    def get_sorted_indexes(self, 
                           values,
                           ascending=True) :
        """Get the indexes to sort the list of values
        
        :param values: list of values to argsort
        :type values: list
        :param ascending: True if sorting needs to be ascending, False for 
            descending order
        :type ascending: bool
        """
        
        values = np.array(values)
        
        if not ascending :
            values = -values
            
        sorted_indexes = values.argsort()
        
        return sorted_indexes
    
    def filter_subset(self,
                      poses,
                      values,
                      ascending=True):
        """Filter a list of poses based on top values. Default will take ratio
        of poses, unless number is defined in pose selector
        
        :param poses: list of poses to filter
        :type poses: list[Poses]
        :param values: values corresponding to each pose
        :type values: list[float]
        :param ascending: True if values sorting needs to be ascending, 
            False for descending order
        :type ascending: bool
        """
        
        sorted_indexes = self.get_sorted_indexes(values=values,
                                                 ascending=ascending)
            
        if self.number is not None:
            assert self.number <= len(values), \
            f'Requested number {self.number} must be lower or equal than number of values {len(values)}'
            limit = self.number
        else :
            limit = int(len(values) * self.ratio)
        selected_indexes = sorted_indexes[:limit]
        poses_subset = [poses[i]
                        for i in selected_indexes]
        
        return poses_subset
        
        
class RandomPoseSelector(PoseSelector) :
    """Rank poses according to a random list of values"""
    
    def __init__(self,
                 selector_name: str='random',
                 ratio: int=0.2,
                 number=None):
        super().__init__(selector_name, ratio, number)
        
    def select_poses(self,
                     poses):
        n_poses = len(poses)
        random_values = np.random.randn((n_poses))
        poses_subset = self.filter_subset(poses, values=random_values)
            
        return poses_subset
  
class ScorePoseSelector(PoseSelector):
    """Rank poses according to docking score (pose.fitness())"""
    
    def __init__(self, 
                 selector_name: str='score',
                 ratio: int=0.2,
                 number=None):
        super().__init__(selector_name, ratio, number)
    
    def select_poses(self,
                     poses,):
        scores = [pose.fitness() for pose in poses]
        poses_subset = self.filter_subset(poses, 
                                          values=scores, 
                                          ascending=False)
            
        return poses_subset
      
      
class RDKitMolRanker() :
    
    def poses_to_rdkit_mol(self, 
                           poses) :
        first_pose = poses[0]
        if first_pose.backend == 'ccdc' :
            rdkit_mol, new_conf_ids = self.ccdc_rdkit_connector.ccdc_ensemble_to_rdkit_mol(
                ccdc_ensemble=poses
                )
        elif first_pose.backend == 'rdkit' :
            rdkit_mol = copy.deepcopy(first_pose.molecule)
            for pose in poses[1:] :
                rdkit_mol.AddConformer(pose.molecule.GetConformer(), 
                                        assignId=True)
        return rdkit_mol
     
class EnergyPoseSelector(PoseSelector, RDKitMolRanker):
    """Rank poses according to their energy (computed in mol featurizer)
    
    :param mol_featurizer: object used to featurize molecule for neural networks
    :type mol_featurizer: MoleculeFeaturizer
    """
    
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
        poses_subset = None
        try :
            rdkit_mol = self.poses_to_rdkit_mol(poses)
            try :
                data_list = self.mol_featurizer.featurize_mol(rdkit_mol)
                energies = [data.energy for data in data_list]
                poses_subset = self.filter_subset(poses, values=energies)
            except AttributeError :
                print('Energy computation bugging')
        except ConversionError :
            print('RDKit to CCDC mol failed')
            
        if poses_subset is None :
            raise PoseSelectionError()
            
        return poses_subset
        
class ModelPoseSelector(PoseSelector, RDKitMolRanker):
    """Rank values according to their predicted RMSD values (model predictions)
    
    :param model: RMSD prediction model (trained)
    :type model: LitSchNet
    :param mol_featurizer: object used to featurize molecule for neural networks
    :type mol_featurizer: MoleculeFeaturizer
    """
    
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
        poses_subset = None
        try :
            rdkit_mol = self.poses_to_rdkit_mol(poses)
            try :
                data_list = self.mol_featurizer.featurize_mol(rdkit_mol)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.model.device)
                    
                with torch.no_grad() :
                    preds = self.model(batch).cpu().numpy()
                    preds = preds.reshape(-1)
                
                poses_subset = self.filter_subset(poses, values=preds)
                
            except AttributeError :
                print('Energy computation bugging')
        except ConversionError :
            print('RDKit to CCDC mol failed')
            
        if poses_subset is None :
            raise PoseSelectionError()
            
        return poses_subset