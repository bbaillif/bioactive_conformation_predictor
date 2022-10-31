from abc import abstractmethod
from rdkit.Chem.rdchem import Mol
from typing import Sequence, Any

class MolFeaturizer():
    
    @abstractmethod
    def featurize_mol(self,
                      mol: Mol) -> Sequence[Any]:
        pass