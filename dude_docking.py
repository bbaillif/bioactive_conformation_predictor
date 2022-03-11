import os
from re import sub
import sys
import numpy as np
import torch
import argparse
import json

from tqdm import tqdm
from rdkit import Chem
from rdkit.ML.Scoring.Scoring import CalcEnrichment, CalcBEDROC
from ccdc.io import MoleculeReader
from ccdc.conformer import ConformerGenerator
from molecule_featurizer import MoleculeFeaturizer
from litschnet import LitSchNet
from gold_docker import GOLDDocker
from pose_reader import PoseReader
from pose_selector import (Pose,
                           RandomPoseSelector,
                           ScorePoseSelector,
                           EnergyPoseSelector,
                           ModelPoseSelector)
from multiprocessing import Pool
from ccdc_rdkit_connector import CcdcRdkitConnector
from collections import defaultdict

class DUDEDocking() :
    """ Performs docking for actives and decoys for a target in DUD-E
    Generated 20 poses per molecule, and compares virtual screening performances
    for different selectors (model, energy, random...)
    
    :param target: Target to dock
    :type target: str
    :param dude_dir: Directory path to DUD-E data
    :type dude_dir: str
    """
    
    def __init__(self,
                 target: str='jak2',
                 dude_dir: str='/home/benoit/DUD-E/all',
                 rigid_docking: bool=False):
        
        self.target = target
        self.dude_dir = dude_dir
        self.rigid_docking = rigid_docking
        
        self.target_dir = os.path.join(self.dude_dir, self.target)
        self.actives_file = 'actives_final.mol2.gz'
        self.actives_path = os.path.join(self.target_dir, self.actives_file)
        self.decoys_file = 'decoys_final.mol2.gz'
        self.decoys_path = os.path.join(self.target_dir, self.decoys_file)
        self.protein_file = 'receptor.pdb'
        self.protein_path = os.path.join(self.target_dir, self.protein_file)
        self.ligand_file = 'crystal_ligand.mol2'
        self.ligand_path = os.path.join(self.target_dir, self.ligand_file)
        
        self.experiment_id = f'{self.target}'
        if self.rigid_docking :
            self.experiment_id = self.experiment_id + '_rigid'
            
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
        
        
    def dock(self) :
        """Performs docking for actives and decoy for setup target
        
        """
        active_mols_reader = MoleculeReader(self.actives_path)
        active_mols = [mol for mol in active_mols_reader]
        active_mol_ids = [f'active_{i}' for i, mol in enumerate(active_mols)]
        
        decoy_mols_reader = MoleculeReader(self.decoys_path)
        decoy_mols = [mol for mol in decoy_mols_reader]
        decoy_mol_ids = [f'decoy_{i}' for i, mol in enumerate(decoy_mols)]
        
        all_mols = active_mols + decoy_mols
        mol_ids = active_mol_ids + decoy_mol_ids
        
        self.gold_docker = GOLDDocker(protein_path=self.protein_path,
                                      native_ligand_path=self.ligand_path,
                                      experiment_id=self.experiment_id)
        
        for mol_id, ccdc_mol in zip(mol_ids, all_mols) :
            try :
                results = self.gold_docker.dock_molecule(ccdc_mol=ccdc_mol,
                                                         mol_id=mol_id,
                                                         n_poses=1,
                                                         rigid=self.rigid_docking)
            except KeyboardInterrupt :
                sys.exit(0)
            except :
                print(f'Docking failed for {mol_id}')
                
                
    def dock_pool(self) :
        
        active_mols_reader = MoleculeReader(self.actives_path)
        n_actives = len(active_mols_reader)
        active_mols = [mol for mol in active_mols_reader]
        active_mol_ids = [f'active_{i}' for i in range(n_actives)]
        
        decoy_mols_reader = MoleculeReader(self.decoys_path)
        n_decoys = len(decoy_mols_reader)
        decoy_mols = [mol for mol in decoy_mols_reader]
        decoy_mol_ids = [f'decoy_{i}' for i in range(n_decoys)]
        
        all_mols = active_mols + decoy_mols
        mol_ids = active_mol_ids + decoy_mol_ids
        
        rdkit_mols = [self.ccdc_rdkit_connector.ccdc_mol_to_rdkit_mol(ccdc_mol=ccdc_mol)
                      for ccdc_mol in all_mols]
        
        print(f'Number of threads : {len(mol_ids)}')
        params = [(mol_id, rdkit_mol) 
                  for mol_id, rdkit_mol 
                  in zip(mol_ids, rdkit_mols)]
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.dock_thread, params)
             
             
    def get_ccdc_mol(self,
                     mol_id) :
        """Does not work with threads since MoleculeReader is using an external
        file that cannot be read multiple times simultaneously"""
        splits = mol_id.split('_')
        dataset = splits[0]
        idx = splits[1]
        if dataset == 'active' :
            path = self.actives_path
        else :
            path = self.decoys_path
        mol_reader = MoleculeReader(path)
        return mol_reader[idx]
                
                
    def dock_thread(self,
                    params) :
        #ccdc_mol = self.get_ccdc_mol(mol_id=mol_id)
        mol_id, rdkit_mol = params
        try :
            ccdc_mol = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_mol)
            
            if self.rigid_docking :
                ccdc_mols = self.generate_conformers_for_molecule(ccdc_mol)
            else :
                ccdc_mols = [ccdc_mol]
            
            self.gold_docker = GOLDDocker(protein_path=self.protein_path,
                                        native_ligand_path=self.ligand_path,
                                        experiment_id=self.experiment_id) 
            results = self.gold_docker.dock_molecules(ccdc_mols=ccdc_mols,
                                                    mol_id=mol_id,
                                                    n_poses=1,
                                                    rigid=self.rigid_docking)
        except :
            print(f'Docking failed for {mol_id}')
               
           
    def generate_conformers_pool(self,
                                 ccdc_mols) :
        
        print(f'Number of threads : {len(ccdc_mols)}')
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.dock_thread, ccdc_mols)
        
        
    def generate_conformers_thread(self,
                                   ccdc_mol) :
        
        conformers = self.generate_conformers_for_molecule(ccdc_mol=ccdc_mol)
        return [conf.molecule for conf in conformers]
    
    
    def generate_conformers_for_molecule(self,
                                         ccdc_mol) :
        conf_generator = ConformerGenerator()
        conf_generator.settings.max_conformers = 100
        conformers = conf_generator.generate(ccdc_mol)
        return [conf.molecule for conf in conformers]
           
                
    def ef_analysis(self,
                    use_cuda=True) :
        """Analyse virtual screening results depending on the ranker used
        to select each molecule pose. Evaluated rankers are model (DL model
        for bioactive RMSD prediction), energy, score and random.
        
        """
        
        self.gold_docker = GOLDDocker(protein_path=self.protein_path,
                                      native_ligand_path=self.ligand_path,
                                      experiment_id=self.experiment_id)
        self.mol_featurizer = MoleculeFeaturizer()
        
        self.model_checkpoint_dir = os.path.join('lightning_logs',
                                                  'random_split_0_new',
                                                  'checkpoints')
        self.model_checkpoint_name = os.listdir(self.model_checkpoint_dir)[0]
        self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir,
                                                  self.model_checkpoint_name)
        self.model = LitSchNet.load_from_checkpoint(self.model_checkpoint_path)
        self.model.eval()
        
        if use_cuda and torch.cuda.is_available() :
            self.model = self.model.to('cuda')
        else :
            self.model = self.model.to('cpu')
        
        self.pose_selectors = {
            'random' : RandomPoseSelector(number=1),
            'score' : ScorePoseSelector(number=1),
            'energy' : EnergyPoseSelector(mol_featurizer=self.mol_featurizer,
                                         number=1),
            'model' : ModelPoseSelector(model=self.model,
                                        mol_featurizer=self.mol_featurizer,
                                        number=1)
        }
        
        dude_docking_dir = os.path.join(self.gold_docker.output_dir,
                                        self.gold_docker.experiment_id)
        docked_dirs = os.listdir(dude_docking_dir)
        active_dirs = [os.path.join(dude_docking_dir, d)
                       for d in docked_dirs 
                       if 'active' in d]
        decoy_dirs = [os.path.join(dude_docking_dir, d) 
                      for d in docked_dirs 
                      if 'decoy' in d]
        
        self.active_selected_poses = defaultdict(list)
        for active_dir in tqdm(active_dirs) :
            self.load_and_select_poses(directory=active_dir,
                                       subset='active')
                
        self.decoy_selected_poses = defaultdict(list)
        for decoy_dir in tqdm(decoy_dirs) :
            self.load_and_select_poses(directory=decoy_dir,
                                       subset='decoy')
               
        results = {}
        for selector_name, pose_selector in self.pose_selectors.items() :
            results[selector_name] = {}
            active_selected_poses = self.active_selected_poses[selector_name]
            decoy_selected_poses = self.decoy_selected_poses[selector_name]
                    
            active_2d_list = [[pose.fitness(), True] 
                              for pose in active_selected_poses]
            decoy_2d_list = [[pose.fitness(), False] 
                             for pose in decoy_selected_poses]
            
            all_2d_list = active_2d_list + decoy_2d_list
            all_2d_array = np.array(all_2d_list)
            
            sorting = np.argsort(-all_2d_array[:, 0])
            sorted_2d_array = all_2d_array[sorting]

            results[selector_name]['ef'] = {}
            fractions = np.arange(start=0.01, 
                                  stop=1.01, 
                                  step=0.01)
            fractions = np.around(fractions, decimals=2)
            efs = CalcEnrichment(sorted_2d_array, 
                                col=1,
                                fractions=fractions)
            efs = np.around(efs, decimals=3)
            for fraction, ef in zip(fractions, efs) :
                print(f'{selector_name} has EF{fraction} of {ef}')
                results[selector_name]['ef'][fraction] = ef
            
            bedroc = CalcBEDROC(sorted_2d_array, 
                                col=1, 
                                alpha=20)
            bedroc = np.around(bedroc, decimals=3)
            print(f'{selector_name} has BEDROC of {bedroc}')
            
            results[selector_name]['all_2d_list'] = all_2d_list
            results[selector_name]['bedroc'] = bedroc
            
        results_path = os.path.join(self.gold_docker.settings.output_directory,
                                    'results.json')
        print(results_path)
        with open(results_path, 'w') as f :
            json.dump(results, f)
        

    def get_poses(self, 
                  directory) :
        """Obtain poses for a given directory (referring to a molecule)
        
        :param directory: directory for GOLD docking of a single molecule
        :type directory: str
        :return: list of poses
        :rtype: list[Pose]
        """
        
        poses = None
        docked_ligands_path = os.path.join(directory,
                                           self.gold_docker.docked_ligand_name)
        if os.path.exists(docked_ligands_path) :
            # poses_reader = Docker.Results.DockedLigandReader(docked_ligands_path,
            #                                                  settings=None)
            # poses_reader = EntryReader(docked_ligands_path)
            # previous solutions are slow or error prone
            try :
                poses_reader = PoseReader(docked_ligands_path)
                poses = [Pose(pose) for pose in poses_reader]
            except :
                print(f'Reading poses failed for {docked_ligands_path}')
        
        return poses
        
        
    def load_and_select_poses(self,
                              directory,
                              subset: str='active') :
        assert subset in ['active', 'decoy']
        poses = self.get_poses(directory=directory)
        if poses :
            for selector_name, pose_selector in self.pose_selectors.items() :
                selected_poses = pose_selector.select_poses(poses)
                selected_pose = selected_poses[0]
                if selected_poses :
                    if subset == 'active' :
                        self.active_selected_poses[selector_name].append(selected_pose)
                    elif subset == 'decoy' :
                        self.decoy_selected_poses[selector_name].append(selected_pose)

            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Perform docking of a DUD-E target')
    parser.add_argument('--target', 
                        type=str, 
                        default='jak2',
                        help='Target to dock')
    # parser.add_argument('--split_name', 
    #                     type=str, 
    #                     default='random',
    #                     help='Split to use for the model and pdbbind test dataset')
    # parser.add_argument('--split_i', 
    #                     type=int, 
    #                     default=0,
    #                     help='Split number to use for model and test set')
    args = parser.parse_args()
    
    # dude_docking = DUDEDocking(target=args.target)
    dude_docking = DUDEDocking(target=args.target,
                               rigid_docking=True)
    #dude_docking.dock_pool()
    dude_docking.ef_analysis()
    
# TODO : iterate over targets
# TODO : store results in jsons for each target
    
# JAK2
# model ef5 = 6.34 bedroc = 0.36
# energy 5.12 0.30
# score 6.21 0.35
# random 5.54 0.28

# DRD3
# model ef5 = 1.44 bedroc 0.18
# energy 1.91 bedroc 0.19
# score 1.43 0.18
# random 2.07 0.20