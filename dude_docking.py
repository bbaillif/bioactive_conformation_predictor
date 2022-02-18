import os
import sys
import numpy as np
import pickle
import torch

from tqdm import tqdm
from rdkit import Chem
from rdkit.ML.Scoring.Scoring import CalcEnrichment, CalcBEDROC
from ccdc.io import MoleculeReader
from molecule_featurizer import MoleculeFeaturizer
from litschnet import LitSchNet
from ccdc.docking import Docker
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
        
        self.gold_docker = GOLDDocker(protein_path=self.protein_path,
                                 native_ligand_path=self.ligand_path,
                                 experiment_id=self.target)
        
        self.mol_featurizer = MoleculeFeaturizer()
        
        self.model_checkpoint_dir = os.path.join('lightning_logs',
                                                  'random_split_0_new',
                                                  'checkpoints')
        self.model_checkpoint_name = os.listdir(self.model_checkpoint_dir)[0]
        self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir,
                                                  self.model_checkpoint_name)
        self.model = LitSchNet.load_from_checkpoint(self.model_checkpoint_path)
        self.model.eval()
        
        if torch.cuda.is_available() :
            self.model = self.model.to('cuda')
        
        self.pose_selectors = {
            'random' : RandomPoseSelector(number=1),
            'score' : ScorePoseSelector(number=1),
            'energy' : EnergyPoseSelector(mol_featurizer=self.mol_featurizer,
                                         number=1),
            'model' : ModelPoseSelector(model=self.model,
                                        mol_featurizer=self.mol_featurizer,
                                        number=1)
        }
        
    def dock(self) :
        active_mols = MoleculeReader(self.actives_path)
        decoy_mols = MoleculeReader(self.decoys_path)
        
        for i, ccdc_mol in enumerate(active_mols) :
            mol_id = f'active_{i}'
            try :
                results = self.gold_docker.dock_molecule(ccdc_mol=ccdc_mol,
                                        mol_id=mol_id)
            except KeyboardInterrupt :
                sys.exit(0)
            except :
                print(f'Docking failed for {mol_id}')
            
        for i, ccdc_mol in enumerate(decoy_mols) :
            mol_id = f'decoys_{i}'
            try :
                results = self.gold_docker.dock_molecule(ccdc_mol=ccdc_mol,
                                        mol_id=mol_id)
            except KeyboardInterrupt :
                sys.exit(0)
            except :
                print(f'Docking failed for {mol_id}')
                
    def ef_analysis(self) :
        
        active_poses, decoy_poses = self.load_poses()
                
        for selector_name, pose_selector in self.pose_selectors.items() :
            active_selected_poses = []
            for poses in tqdm(active_poses) :
                selected_poses = pose_selector.select_poses(poses)
                if selected_poses :
                    selected_pose = selected_poses[0]
                    active_selected_poses.append(selected_pose)
            decoy_selected_poses = []
            for poses in tqdm(decoy_poses) :
                selected_poses = pose_selector.select_poses(poses)
                if selected_poses :
                    selected_pose = selected_poses[0]
                    decoy_selected_poses.append(selected_pose)
                    
            active_2d_list = [[pose.fitness(), True] 
                              for pose in active_selected_poses]
            decoy_2d_list = [[pose.fitness(), False] 
                             for pose in decoy_selected_poses]
            
            all_2d_list = active_2d_list + decoy_2d_list
            all_2d_array = np.array(all_2d_list)
            
            sorting = np.argsort(-all_2d_array[:, 0])
            sorted_2d_array = all_2d_array[sorting]

            ef = CalcEnrichment(sorted_2d_array, 
                                col=1,
                                fractions=[0.05])[0]
            print(f'{selector_name} has EF5% of {ef}')
            
            bedroc = CalcBEDROC(sorted_2d_array, 
                                col=1, 
                                alpha=20)
            print(f'{selector_name} has BEDROC of {bedroc}')
        

    def get_poses(self, directory) :
        
        poses = None
        docked_ligands_path = os.path.join(directory,
                                           self.gold_docker.docked_ligand_name)
        if os.path.exists(docked_ligands_path) :
            poses_reader = Docker.Results.DockedLigandReader(docked_ligands_path,
                                                             settings=None)
            poses = [pose for pose in poses_reader]
        
        return poses
        
        
    def load_poses(self) :
        dude_docking_dir = os.path.join(self.gold_docker.output_dir,
                                        self.gold_docker.experiment_id)
        docked_dirs = os.listdir(dude_docking_dir)
        active_dirs = [os.path.join(dude_docking_dir, d)
                       for d in docked_dirs 
                       if 'active' in d]
        decoy_dirs = [os.path.join(dude_docking_dir, d) 
                      for d in docked_dirs 
                      if 'decoy' in d]
        
        active_poses = []
        for active_dir in tqdm(active_dirs) :
            poses = self.get_poses(active_dir)
            if poses :
                active_poses.append(poses)
                
        decoy_poses = []
        for decoy_dir in tqdm(decoy_dirs) :
            poses = self.get_poses(decoy_dir)
            if poses :
                decoy_poses.append(poses)
                
        return active_poses, decoy_poses
            
if __name__ == '__main__':
    dude_docking = DUDEDocking()
    dude_docking.dock()
    dude_docking.ef_analysis()
    
# TODO : iterate over targets
# TODO : store results in jsons for each target
    
# JAK2
# model ef5 = 6.34 bedroc = 0.36
# energy 5.12 0.30
# score 6.21 0.35
# random 5.54 0.28