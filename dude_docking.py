import os

from ccdc.io import MoleculeReader
from molecule_featurizer import MoleculeFeaturizer
from litschnet import LitSchNet
from gold_docker import GOLDDocker
from pose_selector import (RandomPoseSelector,
                           ScorePoseSelector,
                           EnergyPoseSelector,
                           ModelPoseSelector)

class DUDEDocking() :
    
    def __init__(self,
                 target='jak2',
                 dude_dir='/home/benoit/DUD-E/all'):
        self.target = target
        self.dude_dir = dude_dir
        
        self.target_dir = os.path.join(self.dude_dir, self.target)
        self.actives_file = 'actives_final.mol2.gz'
        self.actives_path = os.path.join(self.target_dir, self.actives_file)
        self.decoys_file = 'decoys_final.mol2.gz'
        self.decoys_path = os.path.join(self.target_dir, self.decoys_file)
        self.protein_file = 'receptor.pdb'
        self.protein_path = os.path.join(self.target_dir, self.protein_file)
        self.ligand_file = 'crystal_ligand.mol2'
        self.ligand_path = os.path.join(self.target_dir, self.ligand_file)
        
        self.mol_featurizer = MoleculeFeaturizer()
        
        self.model_checkpoint_dir = os.path.join('lightning_logs',
                                                  'random_split_0_new',
                                                  'checkpoints')
        self.model_checkpoint_name = os.listdir(self.model_checkpoint_dir)[0]
        self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir,
                                                  self.model_checkpoint_name)
        self.model = LitSchNet.load_from_checkpoint(self.model_checkpoint_path)
        self.model.eval()
        
        self.pose_selectors = []
        self.pose_selectors.append(RandomPoseSelector(number=1))
        self.pose_selectors.append(ScorePoseSelector(number=1))
        self.pose_selectors.append(EnergyPoseSelector(mol_featurizer=self.mol_featurizer,
                                                      number=1))
        self.pose_selectors.append(ModelPoseSelector(model=self.model,
                                                     mol_featurizer=self.mol_featurizer,
                                                     number=1))
        
    def dock(self) :
        active_mols = MoleculeReader(self.actives_path)
        decoy_mols = MoleculeReader(self.decoys_path)
        
        gold_docker = GOLDDocker(protein_path=self.protein_path,
                                 native_ligand_path=self.ligand_path,
                                 experiment_id=self.target)
        
        for i, ccdc_mol in enumerate(active_mols) :
            mol_id = f'active_{i}'
            results = gold_docker.dock_molecule(ccdc_mol=ccdc_mol,
                                      mol_id=mol_id)
            
        for i, ccdc_mol in enumerate(decoy_mols) :
            mol_id = f'decoys_{i}'
            results = gold_docker.dock_molecule(ccdc_mol=ccdc_mol,
                                      mol_id=mol_id)
            
if __name__ == '__main__':
    dude_docking = DUDEDocking()
    dude_docking.dock()