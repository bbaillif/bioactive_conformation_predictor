from bioconfpred.model import ComENetModel
from bioconfpred.ranker import ModelRanker
from bioconfpred.params import COMENET_CONFIG
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
config = COMENET_CONFIG
config['data_split'] = None
checkpoint_path = "/home/bb596/hdd/conf_ranking/data/lightning_logs/random_split_0_ComENetModel/checkpoints/epoch=12-step=61971.ckpt"
model = ComENetModel.load_from_checkpoint(checkpoint_path, 
                                          config=config)
ranker = ModelRanker(model)
mol = Chem.MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O')
mol = Chem.AddHs(mol, addCoords=True)
EmbedMultipleConfs(mol, 250)
ranks = ranker.rank_molecule(mol)
print(ranks)