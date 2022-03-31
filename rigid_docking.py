import os
import numpy as np
import pandas as pd
import torch
import json
import time
import argparse
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem # must be called before ccdc import
from conf_ensemble_library import ConfEnsembleLibrary
from tqdm import tqdm
from pdbbind_metadata_processor import PDBBindMetadataProcessor
from platinum_processor import PlatinumProcessor
from ccdc_rdkit_connector import CcdcRdkitConnector

from collections import defaultdict
from ccdc.io import EntryReader
from molecule_featurizer import MoleculeFeaturizer
from litschnet import LitSchNet
from ccdc.descriptors import MolecularDescriptors
from gold_docker import GOLDDocker
from pose_selector import (Pose, 
                           PoseSelectionError, 
                           RandomPoseSelector,
                           ScorePoseSelector,
                           EnergyPoseSelector,
                           ModelPoseSelector)
from pose_reader import PoseReader
from multiprocessing import Pool, TimeoutError


class RigidDocking() :
    
    def __init__(self,
                 dataset: str = 'pdbbind',
                 split_name: str = 'random',
                 split_i: int = 0,
                 output_dir_prefix: str='gold_docking',
                 use_cuda: bool=False,
                 bioactive_rmsd_threshold: float=2):
        
        assert dataset in ['pdbbind', 'platinum'], 'Dataset must be pdbbind or platinum'
        self.dataset = dataset
        self.split_name = split_name
        self.split_i = split_i
        if self.dataset == 'pdbbind' :
            self.output_dir = f'{output_dir_prefix}_{dataset}_{split_name}_{split_i}'
        else :
            self.output_dir = f'{output_dir_prefix}_{dataset}'
        # self.output_dir = os.path.join('/media/benoit/New Volume/data/',
        #                                self.output_dir)
        self.output_dir = os.path.abspath(self.output_dir)
        self.use_cuda = use_cuda
        self.bioactive_rmsd_threshold = bioactive_rmsd_threshold
        
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
        
        self.flexible_mol_id = 'flexible_mol'
        self.rigid_mol_id = 'rigid_confs'
        
        if self.dataset == 'pdbbind' :
            self.data_processor = PDBBindMetadataProcessor()
            self.prepare_protein = False
        elif self.dataset == 'platinum' :
            self.data_processor = PlatinumProcessor()
            self.prepare_protein = True
            
        self.smiles_df = pd.read_csv('data/smiles_df.csv')
        # self.ce_index_df = pd.read_csv('data/conf_ensembles/index.csv', index_col=0)
        
    
    def get_test_pdb_ids(self) :
        if self.dataset == 'pdbbind' :
            if self.split_name == 'protein_similarity' :
                path = os.path.join('data/',
                                    f'{self.split_name}_splits',
                                    f'test_pdb_{self.split_name}_split_{self.split_i}.txt')
                with open(path, 'r') as f:
                    test_pdb_ids = f.readlines()
                    test_pdb_ids = [pdb.strip() for pdb in test_pdb_ids]
            else :
                path = os.path.join('data/',
                                    f'ligand_{self.split_name}_splits',
                                    f'test_smiles_{self.split_name}_split_{self.split_i}.txt')
                with open(path, 'r') as f:
                    test_smiles = f.readlines()
                    test_smiles = [smiles.strip() for smiles in test_smiles]
                filtered_smiles_df = self.smiles_df[(self.smiles_df['smiles'].isin(test_smiles))
                                                    & (self.smiles_df['dataset'] == 'pdbbind')
                                                    & (self.smiles_df['included'])]
                test_pdb_ids = filtered_smiles_df['id'].unique()
                
        elif self.dataset == 'platinum' :
            test_pdb_ids = self.smiles_df[(self.smiles_df['dataset'] == 'platinum') 
                                       & (self.smiles_df['included'])]['id'].unique()
            
        return test_pdb_ids
        
        
    def get_test_smiles(self) :
        if self.dataset == 'pdbbind' :
            if self.split_name == 'protein_similarity' :
                path = os.path.join('data/',
                                    f'{self.split_name}_splits',
                                    f'test_pdb_{self.split_name}_split_{self.split_i}.txt')
                with open(path, 'r') as f:
                    test_pdb_ids = f.readlines()
                    test_pdb_ids = [pdb.strip() for pdb in test_pdb_ids]
                filtered_smiles_df = self.smiles_df[self.smiles_df['id'].isin(test_pdb_ids)]
                test_smiles = filtered_smiles_df['smiles'].values
            else :
                path = os.path.join('data/',
                                    f'{self.split_name}_splits',
                                    f'test_smiles_{self.split_name}_split_{self.split_i}.txt')
                with open(path, 'r') as f:
                    test_smiles = f.readlines()
                    test_smiles = [smiles.strip() for smiles in test_smiles]
        elif self.dataset == 'platinum' :
            test_smiles = self.smiles_df[(self.smiles_df['dataset'] == 'platinum') 
                                       & (self.smiles_df['included'])]['smiles'].unique()
            
        return test_smiles
        
    def dock_molecule_conformations(self, 
                                    ccdc_mols, # represents different conformations
                                    pdb_id) :
        
        protein_path, ligand_pathes = self.data_processor.get_pdb_id_pathes(pdb_id=pdb_id)
        
        for ligand_path in ligand_pathes :
            if self.dataset == 'platinum' :
                experiment_id = os.path.split(ligand_path)[1].split('.')[0]
            else :
                experiment_id = pdb_id
        
            self.gold_docker = GOLDDocker(protein_path=protein_path,
                                        native_ligand_path=ligand_path,
                                        experiment_id=experiment_id,
                                        output_dir=self.output_dir,
                                        prepare_protein=self.prepare_protein)
            
            results = {}
            results['flexible'] = {}
            results['rigid'] = {}
            first_generated_mol = ccdc_mols[0]
            
            # Flexible 
            _, runtime = self.gold_docker.dock_molecule(ccdc_mol=first_generated_mol,
                                                        mol_id=self.flexible_mol_id,
                                                        n_poses=10,
                                                        return_runtime=True)
            results['flexible']['runtime'] = runtime
            
            # Rigid
            _, runtime = self.gold_docker.dock_molecules(ccdc_mols=ccdc_mols,
                                                        mol_id=self.rigid_mol_id,
                                                        n_poses=5,
                                                        rigid=True,
                                                        return_runtime=True)
            results['rigid']['runtime'] = runtime
            results_path = os.path.join(self.output_dir, experiment_id, 'results.json')
            with open(results_path, 'w') as f :
                json.dump(results, f)
        
        
    def get_smiles_for_pdb_id(self,
                              pdb_id) :
        in_dataset = self.smiles_df['dataset'] == self.dataset
        is_included = self.smiles_df['included']
        is_pdb_id = self.smiles_df['pdb_id'] == pdb_id
        filtered_smiles_df = self.smiles_df[in_dataset & is_included & is_pdb_id]
        return filtered_smiles_df['smiles'].values[0]
        
        
    def dock_molecule_pool(self,
                           test_pdb_ids):
        params = []
        
        cel = ConfEnsembleLibrary()
        cel.load_metadata()
        
        already_done_ids = os.listdir(self.output_dir)
        
        print('Prepare conformations for docking')
        for pdb_id in tqdm(test_pdb_ids) :
            if not pdb_id in already_done_ids :
                try :
                    if self.dataset == 'platinum' :
                        pdb_id = pdb_id.split('_')[1].lower()
                    smiles = self.get_smiles_for_pdb_id(pdb_id)
                    ce = cel.load_ensemble_from_smiles(smiles,
                                                    load_dir='generated')
                    rdkit_mol = ce.mol
                    if rdkit_mol.GetNumConformers() == 100 :
                        params.append((rdkit_mol, pdb_id))
                except :
                    print(f'{pdb_id} not included')
                    
        print(f'Number of threads : {len(params)}')
        with Pool(processes=12, maxtasksperchild=1) as pool :
            # pool.map(self.dock_molecule_conformations_thread, params)
            iterator = pool.imap(self.dock_molecule_conformations_thread, params)
            done_looping = False
            while not done_looping:
                try:
                    try :
                        item = iterator.next(timeout=600)
                    except TimeoutError:
                        print("We lacked patience and got a multiprocessing.TimeoutError")

                except StopIteration:
                    done_looping = True
            
            
    def dock_molecule_conformations_thread(self,
                                           params) :
        rdkit_mol, pdb_id = params
        ccdc_mols = [self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_mol,
                                                                        conf_id=conf_id)
                            for conf_id in range(rdkit_mol.GetNumConformers())]
        try :
            self.dock_molecule_conformations(ccdc_mols=ccdc_mols,
                                            pdb_id=pdb_id)
        except Exception as e :
            print(e)
        
        
    def get_top_poses_rigid(self,
                            pdb_id):
        """For multiple ligand docking, return the top pose for each ligand
        (a ligand is a distinct conf of a molecule in the context of rigid 
        docking)
        
        :param pdb_id: PDB_ID of the molecule, useful to retrieve output dir
        :type pdb_id: str
        :return: List of top poses, one per ligand name
        :rtupe: list[ccdc.docking.Docker.Results.DockedLigand]
        """
        
        top_poses = None 
 
        docked_ligand_path = os.path.join(self.output_dir,
                                          pdb_id,
                                          self.rigid_mol_id,
                                          'docked_ligands.mol2')
        if os.path.exists(docked_ligand_path) :
            poses_reader = PoseReader(docked_ligand_path)
            poses = [Pose(pose) for pose in poses_reader]

            top_poses = []
            seen_ligands = []
            for pose in poses :
                identifier = pose.identifier
                lig_num = identifier.split('|')[1]
                if not lig_num in seen_ligands :
                    top_poses.append(pose)
                    seen_ligands.append(lig_num)
                    
        return top_poses
    
    def get_native_ligand(self,
                          pdb_id) :
        
        if self.dataset == 'platinum' :
            conf_prop = 'platinum_id'
        else :
            conf_prop = 'pdbbind_id'
        
        cel = ConfEnsembleLibrary()
        cel.load_metadata()
        
        smiles = self.smiles_df[(self.smiles_df['dataset'] == self.dataset)
                                & (self.smiles_df['id'] == pdb_id)]['smiles'].values[0]
        
        ce = cel.load_ensemble_from_smiles(smiles)
        
        rdkit_native_ligand = None
        for conf in ce.mol.GetConformers() :
            if conf.HasProp(conf_prop) :
                prop = conf.GetProp(conf_prop)
                prop = prop.replace('e+', 'e')
                if prop == pdb_id :
                    rdkit_native_ligand = copy.deepcopy(ce.mol)
                    rdkit_native_ligand.RemoveAllConformers()
                    rdkit_native_ligand.AddConformer(conf, assignId=True)
                    break
        
        if rdkit_native_ligand is None :
            print(f'{smiles} : {pdb_id} not found')
                    
        
        return rdkit_native_ligand
    
    def analyze_pdb_id(self,
                       pdb_id,
                       rdkit_native_ligand) :
        
        ccdc_rdkit_connector = CcdcRdkitConnector()
        native_ligand = ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_native_ligand,
                                                                    conf_id=0)
        
        print(f'Analyzing {pdb_id}')
        results_path = os.path.join(self.output_dir, pdb_id, 'results.json')
        with open(results_path, 'r') as f :
            results = json.load(f)
        
        docked_ligand_path = os.path.join(self.output_dir,
                                            pdb_id,
                                            self.flexible_mol_id,
                                            'docked_ligands.mol2')
        flexible_poses = None
        if os.path.exists(docked_ligand_path) :
            poses_reader = PoseReader(docked_ligand_path)
            flexible_poses = [Pose(pose) for pose in poses_reader]
        
        rigid_poses = self.get_top_poses_rigid(pdb_id)
        if flexible_poses and rigid_poses :
            
            included = True
            for selector_name, pose_selector in self.pose_selectors.items() :
                if included: # if mol_featurizer fails, sorted_poses is None
                    try :
                        sorted_poses = pose_selector.select_poses(poses=rigid_poses)
                        
                        ranker_results = self.evaluate_ranker(poses=sorted_poses,
                                                native_ligand=native_ligand,
                                                ranker_name=selector_name)
                        results['rigid'][selector_name] = ranker_results
                            
                    except PoseSelectionError :
                        print(f'No pose selected for {pdb_id}')
                        included = False
                
            if included :
                ranker_results = self.evaluate_ranker(poses=rigid_poses,
                                        native_ligand=native_ligand,
                                        ranker_name='CCDC')
                results['rigid']['CCDC'] = ranker_results
                
                # Score/RMSD figures for rigid poses
                scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(poses=rigid_poses,
                                                                native_ligand=native_ligand)
                top_score_index = np.negative(scores).argsort()[0]
                results['rigid']['top_score'] = float(scores[top_score_index])
                results['rigid']['ligand_rmsd_top_score'] = float(ligand_rmsds[top_score_index])
                results['rigid']['overlay_rmsd_top_score'] = float(overlay_rmsds[top_score_index])
                
                top_rmsd_index = np.array(ligand_rmsds).argsort()[0]
                results['rigid']['top_rmsd'] = float(ligand_rmsds[top_rmsd_index])
                results['rigid']['ligand_rmsd_top_rmsd'] = float(ligand_rmsds[top_rmsd_index])
                results['rigid']['overlay_rmsd_top_rmsd'] = float(overlay_rmsds[top_rmsd_index])
                
                # Score/RMSD figures for flexible poses
                scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(poses=flexible_poses,
                                                                native_ligand=native_ligand)
                results['flexible']['top_score'] = float(scores[0])
                results['flexible']['ligand_rmsd_top_score'] = float(ligand_rmsds[0])
                results['flexible']['overlay_rmsd_top_score'] = float(overlay_rmsds[0])
                
                ligand_rmsds = np.array(ligand_rmsds)
                ligand_rmsd_sorting = ligand_rmsds.argsort()
                top_ligand_rmsd_index = ligand_rmsd_sorting[0]
                results['flexible']['top_ligand_rmsd_index'] = int(top_ligand_rmsd_index)
                results['flexible']['top_ligand_rmsd'] = float(ligand_rmsds[top_ligand_rmsd_index])
                results['flexible']['score_top_ligand_rmsd'] = float(scores[top_ligand_rmsd_index])
                results['flexible']['overlay_rmsd_top_ligand_rmsd'] = float(overlay_rmsds[0])

                print(f'Save {pdb_id}')
                with open(results_path, 'w') as f :
                    json.dump(results, f)
                    
            else :
                print(f'{pdb_id} not included')
                    
        else :
            print(f'No pose docked for {pdb_id}')
    
    
    def prepare_analysis(self) :
        
        self.mol_featurizer = MoleculeFeaturizer()
        if self.dataset == 'pdbbind' :
            if self.split_name == 'protein_similarity' :
                split_name = 'protein'
            else :
                split_name = self.split_name
            self.model_checkpoint_dir = os.path.join('lightning_logs',
                                                    f'{split_name}_split_{self.split_i}_v2',
                                                    'checkpoints')
            self.model_checkpoint_name = os.listdir(self.model_checkpoint_dir)[0]
            self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir,
                                                    self.model_checkpoint_name)
            self.model = LitSchNet.load_from_checkpoint(self.model_checkpoint_path)
            self.model.eval()
            
            if self.use_cuda and torch.cuda.is_available() :
                self.model = self.model.to('cuda')
            
            # ratio = 1 for each selector simply ranks the poses according to the
            # selection scheme
            self.pose_selectors = {
                'model' : ModelPoseSelector(model=self.model,
                                            mol_featurizer=self.mol_featurizer,
                                            ratio=1),
                'random' : RandomPoseSelector(ratio=1),
                'score' : ScorePoseSelector(ratio=1),
                'energy' : EnergyPoseSelector(mol_featurizer=self.mol_featurizer,
                                            ratio=1)
            }
            
        else :
            self.pose_selectors = {}
            for split_name in ['random', 'scaffold', 'protein'] :
                self.model_checkpoint_dir = os.path.join('lightning_logs',
                                                    f'{split_name}_split_{self.split_i}_v2',
                                                    'checkpoints')
                self.model_checkpoint_name = os.listdir(self.model_checkpoint_dir)[0]
                self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir,
                                                        self.model_checkpoint_name)
                self.model = LitSchNet.load_from_checkpoint(self.model_checkpoint_path)
                self.model.eval()
                
                if self.use_cuda and torch.cuda.is_available() :
                    self.model = self.model.to('cuda')
                    
                self.pose_selectors[f'model_{split_name}'] = ModelPoseSelector(model=self.model,
                                                                        mol_featurizer=self.mol_featurizer,
                                                                        ratio=1)
            self.pose_selectors['random'] = RandomPoseSelector(ratio=1)
            self.pose_selectors['score'] = ScorePoseSelector(ratio=1)
            self.pose_selectors['energy'] = EnergyPoseSelector(mol_featurizer=self.mol_featurizer,
                                                               ratio=1)
                
    
    
    def docking_analysis_pool(self,
                              single=False):
        
        self.prepare_analysis()
            
        params = []
        pdb_ids = os.listdir(self.output_dir)
        for pdb_id in tqdm(pdb_ids) :
            try :
                rdkit_native_ligand = self.get_native_ligand(pdb_id)
                params.append((pdb_id, rdkit_native_ligand))
            except Exception as e :
                print(f'{pdb_id} failed')
                print(str(e))
            
        print(f'Number of threads : {len(params)}')
        if single :
            for p in params :
                self.docking_analysis_thread(params=p)
        else :
            with Pool(processes=12, maxtasksperchild=1) as pool :
                pool.map(self.docking_analysis_thread, params)
    
    
    def docking_analysis_thread(self, 
                                params):
        
        pdb_id, rdkit_native_ligand = params
        try :
            self.analyze_pdb_id(pdb_id, rdkit_native_ligand)
        except Exception as e :
            print(f'Evaluation failed for {pdb_id}')
            print(str(e))
    
                
    def evaluate_ranker(self, 
                        poses, 
                        native_ligand,
                        ranker_name):
        ranker_results = {}
        scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(poses=poses,
                                                                  native_ligand=native_ligand)
        ligand_rmsds = np.array(ligand_rmsds)
        overlay_rmsds = np.array(overlay_rmsds)
        score_argsort = np.negative(scores).argsort()
        rmsd_argsort = ligand_rmsds.argsort()
        overlay_rmsd_argsort = overlay_rmsds.argsort()
        ranker_results['top_score_index'] = int(score_argsort[0])
        ranker_results['top_rmsd_index'] = int(rmsd_argsort[0])
        ranker_results['top_overlay_rmsd_index'] = int(overlay_rmsd_argsort[0])
        
        bioactive_pose_indexes = np.where(ligand_rmsds <= self.bioactive_rmsd_threshold)[0]
        if len(bioactive_pose_indexes) :
            ranker_results['bioactive_pose_first_index'] = int(bioactive_pose_indexes[0])
            
        bioactive_conf_indexes = np.where(overlay_rmsds <= self.bioactive_rmsd_threshold)[0]
        if len(bioactive_conf_indexes) :
            ranker_results['bioactive_conf_first_index'] = int(bioactive_conf_indexes[0])
        
        return ranker_results
                
    def evaluate_poses(self,
                      poses, 
                      native_ligand):
        
        scores = []
        ligand_rmsds = []
        overlay_rmsds = []
        for pose in poses :
            mol = pose.molecule
            #mol.remove_atoms([atom for atom in mol.atoms if atom.atomic_number < 2])
            scores.append(pose.fitness())
            ccdc_mol = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=mol)
            ligand_rmsds.append(MolecularDescriptors.rmsd(native_ligand, 
                                                          ccdc_mol))
            overlay_rmsds.append(MolecularDescriptors.rmsd(native_ligand, 
                                                           ccdc_mol,
                                                           overlay=True))
        return scores, ligand_rmsds, overlay_rmsds
        
    
    def analysis_report(self, 
                        only_good_docking=True) :
        
        pdb_ids = os.listdir(self.output_dir)
        results = []
        for pdb_id in pdb_ids : 
            result_path = os.path.join(self.output_dir, pdb_id, 'results.json')
            if os.path.exists(result_path) :
                with open(result_path, 'r') as f :
                    results.append(json.load(f))
        
        top_indexes = {}
        top_indexes['score'] = defaultdict(list)
        top_indexes['ligand_rmsd'] = defaultdict(list)
        top_indexes['overlay_rmsd'] = defaultdict(list)
        top_indexes['docking_power'] = defaultdict(list)
        top_indexes['correct_conf'] = defaultdict(list)
        
        flexible_docking_power = defaultdict(list)
        flexible_generation_power = defaultdict(list)
        rigid_generation_power = defaultdict(list)
        
        if self.dataset == 'pdbbind' :
            rankers = ['model', 'energy', 'score', 'random', 'CCDC']
        else :
            rankers = ['model_random', 'model_scaffold', 'model_protein',
                       'energy', 'score', 'random', 'CCDC']
        for result in tqdm(results) :
            rigid_result = result['rigid']
            flexible_result = result['flexible']
            
            if 'top_rmsd' in rigid_result :
                
                good_docking = (rigid_result['top_rmsd'] <= self.bioactive_rmsd_threshold) or (flexible_result['ligand_rmsd_top_score'] <= self.bioactive_rmsd_threshold)
                
                if (only_good_docking and good_docking) or not only_good_docking :
                
                    for ranker in rankers :
                        top_indexes['score'][ranker].append(rigid_result[ranker]['top_score_index'])
                        top_indexes['ligand_rmsd'][ranker].append(rigid_result[ranker]['top_rmsd_index'])
                        top_indexes['overlay_rmsd'][ranker].append(rigid_result[ranker]['top_overlay_rmsd_index'])
                        if 'bioactive_pose_first_index' in rigid_result[ranker] :
                            top_indexes['docking_power'][ranker].append(rigid_result[ranker]['bioactive_pose_first_index'])
                        if 'bioactive_conf_first_index' in rigid_result[ranker] :
                            top_indexes['correct_conf'][ranker].append(rigid_result[ranker]['bioactive_conf_first_index'])
            
                    flexible_docking_power['top_score_pose'].append(flexible_result['ligand_rmsd_top_score'] <= self.bioactive_rmsd_threshold)
                    flexible_docking_power['top_ligand_rmsd_pose'].append(flexible_result['top_ligand_rmsd'] <= self.bioactive_rmsd_threshold)
                    flexible_generation_power['top_score_pose'].append(flexible_result['overlay_rmsd_top_score'] <= self.bioactive_rmsd_threshold)
                    rigid_generation_power['top_score_pose'].append(rigid_result['overlay_rmsd_top_score'] <= self.bioactive_rmsd_threshold)
        
        recall_flexible_top_score = np.array(flexible_docking_power['top_score_pose'])
        recall_flexible_top_ligand_rmsd = np.array(flexible_docking_power['top_ligand_rmsd_pose'])
        recall_flexible_overlay_rmsd = np.array(flexible_generation_power['top_score_pose'])
        n_flexible_mols = len(recall_flexible_top_score)
        
        recall_rigid_overlay_rmsd = np.array(rigid_generation_power['top_score_pose'])
        n_rigid_mols = len(recall_rigid_overlay_rmsd)
        
        print('Flexible n_mols')
        print(n_flexible_mols)
        
        print('Flexible docking power top score')
        print(recall_flexible_top_score.sum() / n_flexible_mols)
        
        print('Flexible docking power top pose')
        print(recall_flexible_top_ligand_rmsd.sum() / n_flexible_mols)
        
        print('Flexible generation power')
        print(recall_flexible_overlay_rmsd.sum() / n_flexible_mols)
        
        print('Rigid generation power')
        print(recall_rigid_overlay_rmsd.sum() / n_rigid_mols)
        
        # Produce lineplots
        xlabel = 'Number of input conformations'
        ylabel = 'Recall'
        columns = [xlabel, ylabel, 'metric', 'ranker']
        df = pd.DataFrame()
        thresholds = range(100)
        for metric, top_indexes_metric in top_indexes.items():
            print(metric)
            for ranker, top_indexes_task in top_indexes_metric.items():
                print(ranker)
                top_indexes_task = np.array(top_indexes_task)
                n_rigid_mols = len(top_indexes_task)
            
                recalls = []
                for threshold in thresholds :
                    recalls.append(np.sum(top_indexes_task <= threshold))
                    
                current_df = pd.DataFrame(zip(thresholds, recalls), 
                                          columns=[xlabel, ylabel])
                current_df['metric'] = metric
                current_df['ranker'] = ranker
                df = df.append(current_df, ignore_index=True)
                    
            plot_df = df[df['metric'] == metric]
            
            #import pdb; pdb.set_trace()
            
            sns.lineplot(data=plot_df, x=xlabel, y=ylabel, hue='ranker')
            
            if metric in ['docking_power', 'correct_conf'] :
                # plt.axhline(y=recall_flexible_top_score.sum(), 
                #             label='Flexible docking power top score',
                #             color='grey')
                plt.axhline(y=recall_flexible_top_ligand_rmsd.sum(), 
                            label='Flexible docking power',
                            color='grey')
                plt.axhline(y=n_flexible_mols,
                            label='Total number of molecules',
                            color='black')
                
            if metric == 'docking_power' :
                title = 'Docking power'
            elif metric == 'ligand_rmsd' :
                title = 'Retrieval of best pose'
            else :
                title = f'Retrieval of top {metric}'
            plt.title(title)
            plt.legend()
            plt.ylim(0)
            if only_good_docking :
                suffix = 'successful_only'
            else :
                suffix = 'all'
            
            if self.dataset == 'pdbbind' :
                fig_name = f'retrieval_{metric}_{self.dataset}_{self.split_name}_{suffix}.png'
            else :
                fig_name = f'retrieval_{metric}_{self.dataset}_{suffix}.png'
            save_path = os.path.join('figures/',
                                     fig_name)
            plt.savefig(save_path)
            plt.clf()
           
        # normalize for comparison across models
        df['Recall'] = df['Recall'] / n_flexible_mols
            
        # save recalls
        if self.dataset == 'pdbbind' :
            if self.split_name == 'protein_similarity' :
                split = 'protein'
            else :
                split = self.split_name
            df_path = os.path.join('results/',
                                    f'{split}_split_{self.split_i}_v2_{self.dataset}/',
                                    f'rigid_ligand_docking_recall_{suffix}.csv')
        else :
            df_path = os.path.join('results/',
                                    f'rigid_ligand_docking_recall_{self.dataset}.csv')
        df.to_csv(df_path)
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Perform rigid docking of a dataset')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='pdbbind',
                        help='Dataset to dock')
    parser.add_argument('--split_name', 
                        type=str, 
                        default='random',
                        help='Split to use for the model and pdbbind test dataset')
    parser.add_argument('--split_i', 
                        type=int, 
                        default=0,
                        help='Split number to use for model and test set')
    args = parser.parse_args()
    
    rigid_docking = RigidDocking(dataset=args.dataset,
                                 split_name=args.split_name,
                                 split_i=args.split_i)
        
    conf_ensemble_library = ConfEnsembleLibrary()
    conf_ensemble_library.load_metadata()
  
    test_pdb_ids = rigid_docking.get_test_pdb_ids()

    rigid_docking.dock_molecule_pool(test_pdb_ids=test_pdb_ids)
    
    # start_time = time.time()  
    # rigid_docking.docking_analysis_pool()
    # runtime = time.time() - start_time
    # print(f'{runtime} seconds runtime')

    # rigid_docking.analysis_report()
            