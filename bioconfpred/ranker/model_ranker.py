import torch

from typing import Sequence, List
from .conf_ranker import ConfRanker
from bioconfpred.model import ConfPredModel
from rdkit.Chem import Mol
from torch_geometric.data import Data

class ModelRanker(ConfRanker):
    """Can rank conformers based on ConfPredModel (RMSD) outputs

    :param model: Input trained model
    :type model: ConfPredModel
    :param batch_size: batch size for feeding the test conformers, 
        defaults to 128
    :type batch_size: int, optional
    :param name: Name of the model, defaults to None
    :type name: str, optional
    :param ascending: Set to True to rank conformers by ascending prediction, 
        else it ranks by descending predictions, defaults to True
    :type ascending: bool, optional
    :param use_cuda: Set to True to use GPU for prediction, defaults to True
    :type use_cuda: bool, optional
    """
    
    def __init__(self,
                 model: ConfPredModel,
                 batch_size: int = 128,
                 name: str = None,
                 ascending: bool = True,
                 use_cuda: bool = True) -> None:

        if name is None:
            name = model.name
        super().__init__(name, ascending)
        self.model = model
        self.batch_size = batch_size
        if torch.cuda.is_available() and use_cuda:
            self.model.to('cuda')
        self.model.eval()
    
    def get_input_list_for_mol(self, 
                                mol: Mol) -> List[Data]:
        """Get torch geometric input list from molecule

        :param mol: Input molecule
        :type mol: Mol
        :return: List of torch geometric Data
        :rtype: List[Data]
        """
        input_list = self.model.featurizer.featurize_mol(mol)
        return input_list
    
    def compute_values(self,
                    input_list: List[Data]) -> Sequence[float]:
        """Get predicted values from model based on input Data list

        :param input_list: Input list of torch geometric Data
        :type input_list: List[Data]
        :return: Predicted values
        :rtype: Sequence[float]
        """
        
        torch.cuda.empty_cache()
        with torch.inference_mode():
            values = self.model.get_preds_for_data_list(input_list)
            values = values.cpu()
            return values.numpy().reshape(-1)
    
    
    
    
# class BioSchNetConfRanker(ModelRanker):
    
#     def __init__(self,
#                  model: BioSchNet,
#                  batch_size: int = 128,
#                  name: str = 'BioSchNet',
#                  ascending: bool = True,
#                  use_cuda: bool = True):
#         super().__init__(model, batch_size, name, ascending, use_cuda)
#         self.mol_featurizer = MoleculeFeaturizer()
        
#     def get_input_list_from_mol(self, 
#                                mol: Mol) -> List[Any]:
#         input_list = self.mol_featurizer.featurize_mol(mol)
#         return input_list
        
#     def get_preds(self,
#                   input_list) -> Sequence[float]:
#         data_loader = PyGDataLoader(input_list, 
#                                     batch_size=self.batch_size)
#         preds = None
#         for batch in data_loader:
#             batch.to(self.model.device)
#             pred = self.model(batch)
#             pred = pred.detach().cpu().numpy().squeeze(1)
#             if preds is None:
#                 preds = pred
#             else:
#                 try:
#                     preds = np.concatenate([preds, pred])
#                 except: 
#                     import pdb; pdb.set_trace()
#         return preds
    
#     def get_input_list_from_data_list(self, 
#                                       data_list: Sequence[Any]) -> Sequence[Any]:
#         input_list = data_list
#         return input_list
    
#     def get_output_list_from_data_list(self, 
#                                        data_list: Sequence[Any]) -> Sequence[Any]:
#         batch = Batch.from_data_list(data_list)
#         return batch.rmsd
    
    
# class E3FPConfRanker(ModelConfRanker):
    
#     def __init__(self,
#                  model: E3FPModel,
#                  batch_size: int = 128,
#                  level: int = 5,
#                  name: str = 'E3FPModel',
#                  ascending: bool = True,
#                  use_cuda: bool = True):
#         super().__init__(model, batch_size, name, ascending, use_cuda)
#         self.level = level
        
#     def get_input_list_from_mol(self, 
#                                mol: Mol) -> List[Any]:
#         fp_matrix = self.get_fp_matrix(mol)
#         return fp_matrix
        
#     def get_preds(self, 
#                   input_list) -> Sequence[float]:
#         input_list = [data for data, rmsd in input_list]
#         fp_matrix = torch.vstack(input_list)
#         data_loader = DataLoader(fp_matrix, batch_size=self.batch_size)
#         preds = np.empty((0,))
#         for x in data_loader:
#             x = x.to(self.model.device)
#             pred = self.model(x)
#             pred = pred.detach().cpu().numpy().squeeze()
#             preds = np.vstack([preds, pred])
#         return preds
    
#     def get_fp_matrix(self, 
#                       mol: Mol) -> torch.Tensor:
#         fp_dict = fprints_dict_from_mol(mol, 
#                                         bits=self.model.n_bits, 
#                                         level=self.level, 
#                                         first=-1, 
#                                         counts=True)
#         fps = fp_dict[self.level]
            
#         db = FingerprintDatabase(fp_type=CountFingerprint, 
#                                  level=self.level)
#         db.add_fingerprints(fps)
#         csr_array = self.db.array
#         array = csr_matrix.toarray(csr_array)
#         array = np.int16(array).squeeze()
#         fp = torch.tensor(array, dtype=torch.float32)
#         return fp
    
#     def get_input_list_from_data_list(self, 
#                                       data_list: Sequence[Any]) -> Sequence[Any]:
#         input_list = [data for data, rmsd in data_list]
#         return input_list
    
#     def get_output_list_from_data_list(self, 
#                                       data_list: Sequence[Any]) -> Sequence[Any]:
#         output_list = [rmsd for data, rmsd in data_list]
#         return output_list