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
                 rigid_docking: bool=False,
                 use_selector_scoring: bool=False):
        
        self.target = target
        self.dude_dir = dude_dir
        self.rigid_docking = rigid_docking
        self.use_selector_scoring = use_selector_scoring
        
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
        print(f'Docking {self.target}')
        
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
                                      output_dir='/hdd/gold_docking_dude',
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
                n_poses = 1
            else :
                ccdc_mols = [ccdc_mol]
                n_poses = 20
            
            self.gold_docker = GOLDDocker(protein_path=self.protein_path,
                                        native_ligand_path=self.ligand_path,
                                        output_dir='/hdd/gold_docking_dude',
                                        experiment_id=self.experiment_id) 
            results = self.gold_docker.dock_molecules(ccdc_mols=ccdc_mols,
                                                    mol_id=mol_id,
                                                    n_poses=n_poses,
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
                                      output_dir='/hdd/gold_docking_dude',
                                      experiment_id=self.experiment_id)
        self.mol_featurizer = MoleculeFeaturizer()
        
        self.pose_selectors = {}
        
        self.active_max_scores = [] # [{percentage_conf : max_score}]
        self.decoy_max_scores = []
        
        for split in ['random', 'scaffold', 'protein'] :
        
            model_checkpoint_dir = os.path.join('lightning_logs',
                                                    f'{split}_split_0',
                                                    'checkpoints')
            model_checkpoint_name = os.listdir(model_checkpoint_dir)[0]
            model_checkpoint_path = os.path.join(model_checkpoint_dir,
                                                 model_checkpoint_name)
            model = LitSchNet.load_from_checkpoint(model_checkpoint_path)
            model.eval()
            
            if use_cuda and torch.cuda.is_available() :
                model = model.to('cuda')
            else :
                model = model.to('cpu')
        
            self.pose_selectors[f'model_{split}'] = ModelPoseSelector(model=model,
                                                                    mol_featurizer=self.mol_featurizer,
                                                                    ratio=1)
        
        self.pose_selectors['random'] = RandomPoseSelector(ratio=1)
        self.pose_selectors['score'] = ScorePoseSelector(ratio=1)
        self.pose_selectors['energy'] = EnergyPoseSelector(mol_featurizer=self.mol_featurizer,
                                                            ratio=1)
        
        dude_docking_dir = os.path.join(self.gold_docker.output_dir,
                                        self.gold_docker.experiment_id)
        docked_dirs = os.listdir(dude_docking_dir)
        active_dirs = [os.path.join(dude_docking_dir, d)
                       for d in docked_dirs 
                       if 'active' in d]
        decoy_dirs = [os.path.join(dude_docking_dir, d) 
                      for d in docked_dirs 
                      if 'decoy' in d]
        
        for active_dir in tqdm(active_dirs) :
            max_scores = self.evaluate_molecule(directory=active_dir)
            if max_scores is not None :
                self.active_max_scores.append(max_scores)
                
        for decoy_dir in tqdm(decoy_dirs) :
            max_scores = self.evaluate_molecule(directory=decoy_dir)
            if max_scores is not None :
                self.decoy_max_scores.append(max_scores)
               
        results = {} # selector_name, percentage, efs, fractions
        for selector_name in self.pose_selectors.keys() :
            selector_name_results = {}
            
            active_max_scores = [max_scores[selector_name] 
                                 for max_scores in self.active_max_scores]
            decoy_max_scores = [max_scores[selector_name] 
                                 for max_scores in self.decoy_max_scores]
            for percentage in range(1, 101) :
                percentage_results = {}
                active_2d_list = [[max_scores[percentage], True] 
                                  for max_scores in active_max_scores]
                decoy_2d_list = [[max_scores[percentage], False] 
                                  for max_scores in decoy_max_scores]
            
                all_2d_list = active_2d_list + decoy_2d_list
                all_2d_array = np.array(all_2d_list)
                
                if selector_name == 'score' or not self.use_selector_scoring:
                    sorting = np.argsort(-all_2d_array[:, 0]) # the minus is important here
                elif self.use_selector_scoring :
                    sorting = np.argsort(all_2d_array[:, 0])
                sorted_2d_array = all_2d_array[sorting]

                fractions = np.arange(start=0.01, 
                                    stop=1.01, 
                                    step=0.01)
                fractions = np.around(fractions, decimals=2)
                efs = CalcEnrichment(sorted_2d_array, 
                                    col=1,
                                    fractions=fractions)
                efs = np.around(efs, decimals=3)
                ef_results = {}
                for fraction, ef in zip(fractions, efs) :
                    # print(f'{selector_name} has EF{fraction} of {ef}')
                    ef_results[fraction] = ef
                percentage_results['ef'] = ef_results
                
                bedroc = CalcBEDROC(sorted_2d_array, 
                                    col=1, 
                                    alpha=20)
                bedroc = np.around(bedroc, decimals=3)
                # print(f'{selector_name} has BEDROC of {bedroc}')
                
                percentage_results['all_2d_list'] = all_2d_list
                percentage_results['bedroc'] = bedroc
                
                selector_name_results[percentage] = percentage_results
                
            results[selector_name] = selector_name_results
            
        if self.use_selector_scoring :
            results_file_name = 'results_custom_score.json'
        else :
            results_file_name = 'results_docking_score_percentage.json'
        results_path = os.path.join(self.gold_docker.settings.output_directory,
                                    results_file_name)
        print(results_path)
        with open(results_path, 'w') as f :
            json.dump(results, f)
        
        
    def evaluate_molecule(self,
                          directory,
                          subset: str='active') :
        max_scores = {}
        assert subset in ['active', 'decoy']
        poses = self.get_poses(directory=directory)
        if poses :
            included = True
            for selector_name, pose_selector in self.pose_selectors.items() :
                ranked_poses = pose_selector.select_poses(poses)
                if ranked_poses and included :
                    max_scores[selector_name] = {}
                    n_poses = len(ranked_poses)
                    for percentage in range(1, 101) :
                        try :
                            n_conf = percentage * n_poses // 100
                            n_conf = max(1, n_conf) # avoid n_conf = 0 situation
                            pose_subset = ranked_poses[:n_conf]
                            subset_scores = [pose.fitness() 
                                            for pose in pose_subset]
                            max_score = np.max(subset_scores)
                            max_scores[selector_name][percentage] = max_score
                        except :
                            import pdb;pdb.set_trace()
                else :
                    max_scores = None
                    included = False
                    break
        else :
            max_scores = None
        return max_scores
    

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
    parser.add_argument('--rigid', 
                        action='store_true',
                        help='Generate conformations for rigid docking, else flexible')
    parser.add_argument('--use_selector_scoring', 
                        action='store_true',
                        help='Whether to replace the score (fitness) of the pose with the selector scoring')
    args = parser.parse_args()
    
    if args.target == 'all' :
        targets = os.listdir('/home/benoit/DUD-E/all')
    else :
        targets = [args.target]
        
    targets = ['bace1']
    for target in targets :
        dude_docking = DUDEDocking(target=target,
                                   rigid_docking=True)
        # dude_docking.dock_pool()
        dude_docking.ef_analysis()
        dude_docking = DUDEDocking(target=target,
                                   rigid_docking=False)
        dude_docking.dock_pool()
        dude_docking.ef_analysis()
        
    # targets = ['jak2']
    # for target in targets :
        # dude_docking = DUDEDocking(target=target,
        #                            rigid_docking=True)
        # dude_docking.dock_pool()
        # dude_docking.ef_analysis()
        # dude_docking = DUDEDocking(target=target,
        #                            rigid_docking=False)
        # dude_docking.dock_pool()
        # dude_docking.ef_analysis()
    
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