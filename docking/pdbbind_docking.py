import os
import numpy as np
import pandas as pd
import json
import time
import copy
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem # must be called before ccdc import
from conf_ensemble import ConfEnsemble
from tqdm import tqdm
from data import PDBBindMetadataProcessor
from ccdc_rdkit_connector import CcdcRdkitConnector

from collections import defaultdict
from model import SchNetModel
from ccdc.descriptors import MolecularDescriptors
from gold_docker import GOLDDocker
from multiprocessing import Pool, TimeoutError
from data.split import MoleculeSplit, ProteinSplit
from rankers import (ModelRanker, 
                     EnergyRanker, 
                     CCDCRanker, 
                     ShuffleRanker, 
                     PropertyRanker,
                     ConfRanker)
from typing import List

class PDBbindDocking() :
    
    def __init__(self,
                 output_dir: str = '/home/bb596/hdd/gold_docking_pdbbind/',
                 data_dir: str = '/home/bb596/hdd/pdbbind_bioactive/data/',
                 use_cuda: bool = False,
                 pose_rmsd_threshold: float = 2,
                 conf_rmsd_threshold: float = 1):

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir) :
            os.mkdir(self.output_dir)
        self.data_dir = data_dir
        self.use_cuda = use_cuda
        self.pose_rmsd_threshold = pose_rmsd_threshold
        self.conf_rmsd_threshold = conf_rmsd_threshold
        
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
        
        self.flexible_mol_id = 'flexible_mol'
        self.rigid_mol_id = 'rigid_confs'
        
        self.data_processor = PDBBindMetadataProcessor(root='/home/bb596/hdd/PDBBind')
        self.prepare_protein = False
            
        self.pdbbind_df = pd.read_csv(os.path.join(data_dir, 'pdbbind_df.csv'))
        self.cel_df = pd.read_csv(os.path.join(data_dir, 'pdb_conf_ensembles', 'ensemble_names.csv'))
        
        self.fractions = np.around(np.arange(0.01, 1.01, 0.01), 2)
    
    def get_test_pdb_ids(self,
                         splits=['random', 'scaffold', 'protein'],
                         split_is=range(5)) :
        test_pdb_ids = []
        for split in splits :
            for split_i in split_is :

                if split == 'protein' :
                    data_split = ProteinSplit(split, split_i)
                else:
                    data_split = MoleculeSplit(split, split_i)
                    
                pdb_ids = data_split.get_pdb_ids('test')
                test_pdb_ids.extend(pdb_ids)
            
        return set(test_pdb_ids)
    
        
    def dock_molecule_conformations(self, 
                                    ccdc_mols, # represents different conformations
                                    pdb_id) :
        
        protein_path, ligand_pathes = self.data_processor.get_pdb_id_pathes(pdb_id=pdb_id)
        
        for ligand_path in ligand_pathes :
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
                                                        n_poses=10,
                                                        rigid=True,
                                                        return_runtime=True)
            results['rigid']['runtime'] = runtime
            results_path = os.path.join(self.output_dir, experiment_id, 'results.json')
            with open(results_path, 'w') as f :
                json.dump(results, f)
                
        return results
        
        
    def get_ce_from_pdb_id(self,
                           pdb_id,
                           library_name='gen_conf_ensembles_moe_all'):
        name = self.pdbbind_df[self.pdbbind_df['pdb_id'] == pdb_id]['ligand_name'].values[0]
        filename = self.cel_df[self.cel_df['ensemble_name'] == name]['filename'].values[0]
        filepath = os.path.join(self.data_dir, library_name, filename)
        ce = ConfEnsemble.from_file(filepath, name=name)
        return ce
        
        
    def dock_molecule_pool(self,
                           test_pdb_ids):
        params = []
        
        print('Prepare conformations for docking')
        for pdb_id in tqdm(test_pdb_ids) :
            flexible_ligands_path = os.path.join(self.output_dir,
                                                 pdb_id,
                                                 'flexible_mol',
                                                 'docked_ligands.mol2')
            rigid_ligands_path = os.path.join(self.output_dir,
                                                 pdb_id,
                                                 'rigid_confs',
                                                 'docked_ligands.mol2')
            if (not os.path.exists(flexible_ligands_path)) or (not os.path.exists(rigid_ligands_path)) :
                try :
                    ce = self.get_ce_from_pdb_id(pdb_id)
                    rdkit_mol = ce.mol
                    params.append((rdkit_mol, pdb_id))
                except Exception as e :
                    print(f'{pdb_id} not included')
                    print(str(e))
                    
        print(f'Number of threads : {len(params)}')
        with Pool(processes=20, maxtasksperchild=1) as pool :
            # pool.map(self.dock_molecule_conformations_thread, params)
            iterator = pool.imap(self.dock_molecule_conformations_thread, params)
            done_looping = False
            while not done_looping:
                try:
                    try:
                        item = iterator.next(timeout=600)
                    except TimeoutError:
                        print("Docking is too long, returning TimeoutError")

                except StopIteration:
                    done_looping = True
            
            
    def dock_molecule_conformations_thread(self,
                                           params) :
        rdkit_mol, pdb_id = params
        ccdc_mols = [self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_mol,
                                                                        conf_id=conf_id)
                            for conf_id in range(rdkit_mol.GetNumConformers())]
        results = None
        try :
            results = self.dock_molecule_conformations(ccdc_mols=ccdc_mols,
                                                       pdb_id=pdb_id)
        except Exception as e :
            print(e)
            
        return results
        
        
    def get_top_poses_rigid(self,
                            pdb_id):
        """For multiple ligand docking, return the top pose for each ligand
        (a ligand is a distinct conf of a molecule in the context of rigid 
        docking)
        
        :param pdb_id: PDB_ID of the molecule, useful to retrieve output dir
        :type pdb_id: str
        :return: List of top poses, one per ligand name
        :rtype: list[ccdc.docking.Docker.Results.DockedLigand]
        """
 
        ce = None
 
        docked_ligand_path = os.path.join(self.output_dir,
                                          pdb_id,
                                          self.rigid_mol_id,
                                          'docked_ligands.mol2')
        if os.path.exists(docked_ligand_path) :
            ce = ConfEnsemble.from_file(docked_ligand_path)

            seen_ligands = []
            for pose in ce.mol.GetConformers() :
                identifier = pose.GetProp('_Name')
                lig_num = identifier.split('|')[1]
                if not lig_num in seen_ligands :
                    seen_ligands.append(lig_num)
                else:
                    conf_id = pose.GetId()
                    ce.mol.RemoveConformer(conf_id)
                    
        return ce
    
    def get_native_ligand(self,
                          pdb_id) :
        
        conf_prop = 'PDB_ID'
        
        ce = self.get_ce_from_pdb_id(pdb_id, 
                                    library_name='pdb_conf_ensembles')
        
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
            print(f'{pdb_id} not found')
        
        return rdkit_native_ligand
    
    
    def prepare_analysis(self,
                         split='random',
                         split_i=0) :
        self.split = split
        self.split_i = split_i
        self.model_checkpoint_dir = os.path.join('lightning_logs',
                                                f'{split}_split_{split_i}',
                                                'checkpoints')
        self.model_checkpoint_name = os.listdir(self.model_checkpoint_dir)[0]
        self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir,
                                                self.model_checkpoint_name)
        config = {"num_interactions": 6,
                  "cutoff": 10,
                  "lr":1e-5,
                  'batch_size': 256,
                  'data_split': MoleculeSplit()}
        model = SchNetModel.load_from_checkpoint(self.model_checkpoint_path, config=config)
        
        self.rankers: List[ConfRanker] = [
            ModelRanker(model=model, use_cuda=self.use_cuda),
            ShuffleRanker(),
            CCDCRanker(),
            EnergyRanker(),
            PropertyRanker(descriptor_name='Gold.PLP.Fitness',
                           ascending=False)
        ]
            
    
    def docking_analysis_pool(self,
                              split,
                              split_i,
                              single=False):
        
        self.prepare_analysis(split=split,
                              split_i=split_i)
            
        pdb_ids = self.get_test_pdb_ids(splits=[split],
                                        split_is=[split_i])
            
        params = []
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
            with Pool(processes=20, maxtasksperchild=1) as pool :
                iterator = pool.imap(self.docking_analysis_thread, params)
                done_looping = False
                while not done_looping:
                    try:
                        try:
                            results = iterator.next(timeout=1200)
                        except TimeoutError:
                            print("Analysis is too long, returning TimeoutError")
                            return 0
                    except StopIteration:
                        done_looping = True
    
    
    
    def docking_analysis_thread(self, 
                                params):
        
        pdb_id, rdkit_native_ligand = params
        results = None
        try :
            results = self.analyze_pdb_id(pdb_id, rdkit_native_ligand)
        except Exception as e :
            print(f'Evaluation failed for {pdb_id}')
            print(str(e))
            
        return results
    
    
    def analyze_pdb_id(self,
                       pdb_id,
                       rdkit_native_ligand) :
        
        ccdc_rdkit_connector = CcdcRdkitConnector()
        native_ligand = ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_native_ligand,
                                                                    conf_id=0)
        
        print(f'Analyzing {pdb_id}')
        results = None
        results_path = os.path.join(self.output_dir, 
                                    pdb_id, 
                                    'results.json')
        with open(results_path, 'r') as f :
            results = json.load(f)
        
        docked_ligand_path = os.path.join(self.output_dir,
                                            pdb_id,
                                            self.flexible_mol_id,
                                            'docked_ligands.mol2')
        flexible_poses = None
        if os.path.exists(docked_ligand_path) :
            ce_flexible_poses = ConfEnsemble.from_file(docked_ligand_path)
        
        ce_rigid_poses = self.get_top_poses_rigid(pdb_id)
        results['n_poses'] = ce_rigid_poses.mol.GetNumConformers()
        if ce_flexible_poses and ce_rigid_poses :
            
            included = True
            rigid_results = {}
            for ranker in self.rankers:
                try :
                    mol = ranker.rank_confs(ce_rigid_poses.mol)
                except Exception as e:
                    print(ranker)
                    print(str(e))
                    print(f'No pose selected for {pdb_id}')
                    included = False
                    break
                else:
                    ranker_results = self.evaluate_ranker(mol, 
                                                          native_ligand)
                    rigid_results[ranker.name] = ranker_results
                    
            if included :
                
                for ranker in self.rankers:
                    results['rigid'][ranker.name] = rigid_results[ranker.name]
                
                # Score/RMSD figures for rigid poses
                scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(ce_rigid_poses.mol,
                                                                          native_ligand)
                top_score_index = np.negative(scores).argsort()[0]
                results['rigid']['top_score'] = float(scores[top_score_index])
                results['rigid']['ligand_rmsd_top_score'] = float(ligand_rmsds[top_score_index])
                results['rigid']['overlay_rmsd_top_score'] = float(overlay_rmsds[top_score_index])
                
                top_rmsd_index = np.array(ligand_rmsds).argsort()[0]
                results['rigid']['score_top_rmsd'] = float(scores[top_rmsd_index])
                results['rigid']['ligand_rmsd_top_rmsd'] = float(ligand_rmsds[top_rmsd_index])
                results['rigid']['overlay_rmsd_top_rmsd'] = float(overlay_rmsds[top_rmsd_index])
                
                top_overlay_rmsd_index = np.array(overlay_rmsds).argsort()[0]
                results['rigid']['score_top_overlay_rmsd'] = float(scores[top_overlay_rmsd_index])
                results['rigid']['ligand_rmsd_top_overlay_rmsd'] = float(ligand_rmsds[top_overlay_rmsd_index])
                results['rigid']['overlay_rmsd_top_overlay_rmsd'] = float(overlay_rmsds[top_overlay_rmsd_index])
                
                # Score/RMSD figures for flexible poses
                scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(ce_flexible_poses.mol,
                                                                          native_ligand)
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
                split_results_path = results_path.replace('results', 
                                                          f'results_{self.split}_{self.split_i}')
                with open(split_results_path, 'w') as f :
                    json.dump(results, f)
                    
            else :
                print(f'{pdb_id} not included')
                    
        else :
            print(f'No pose docked for {pdb_id}')
            
        return results
    
                
    def evaluate_ranker(self, 
                        mol, # has ranked conformations
                        native_ligand):
        ranker_results = {}
        scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(mol,
                                                                  native_ligand)
        ligand_rmsds = np.array(ligand_rmsds)
        overlay_rmsds = np.array(overlay_rmsds)
        
        n_poses = len(ligand_rmsds)
        for fraction in self.fractions:
            fraction_result = {}
            
            n_conf = int(np.ceil(n_poses * fraction))
            score_subset = scores[:n_conf]
            rmsd_subset = ligand_rmsds[:n_conf]
            overlay_subset = overlay_rmsds[:n_conf]
            
            score_argsort = np.negative(score_subset).argsort()
            rmsd_argsort = rmsd_subset.argsort()
            overlay_argsort = overlay_subset.argsort()
            
            top_score_index = score_argsort[0]
            fraction_result['top_score_index'] = int(top_score_index)
            fraction_result['top_score_norm_index'] = int(top_score_index) / n_poses
            
            top_rmsd_index = rmsd_argsort[0]
            fraction_result['top_rmsd_index'] = int(top_rmsd_index)
            fraction_result['top_rmsd_norm_index'] = int(top_rmsd_index) / n_poses
            
            top_overlay_index = overlay_argsort[0]
            fraction_result['top_overlay_rmsd_index'] = int(top_overlay_index)
            fraction_result['top_overlay_rmsd_norm_index'] = int(top_overlay_index) / n_poses
            
            top_score_is_bioactive_pose = rmsd_subset[top_score_index] <= self.pose_rmsd_threshold
            fraction_result['top_score_is_bioactive_pose'] = bool(top_score_is_bioactive_pose)
            
            top_rmsd_is_bioactive_pose = rmsd_subset[top_rmsd_index] <= self.pose_rmsd_threshold
            fraction_result['top_rmsd_is_bioactive_pose'] = bool(top_rmsd_is_bioactive_pose)
            
            top_overlay_is_bioactive_conf = overlay_subset[top_overlay_index] <= self.conf_rmsd_threshold
            fraction_result['top_overlay_is_bioactive_conf'] = bool(top_overlay_is_bioactive_conf)
            
            fraction_result['top_score_rmsd'] = float(rmsd_subset[top_score_index])
            fraction_result['top_rmsd_score'] = float(score_subset[top_rmsd_index])
            
            ranker_results[fraction] = fraction_result
        
        bioactive_pose_indexes = np.where(ligand_rmsds <= self.pose_rmsd_threshold)[0]
        if len(bioactive_pose_indexes) :
            ranker_results['bioactive_pose_first_index'] = int(bioactive_pose_indexes[0])
            ranker_results['bioactive_pose_first_normalized_index'] = int(bioactive_pose_indexes[0]) / n_poses
            
        bioactive_conf_indexes = np.where(overlay_rmsds <= self.conf_rmsd_threshold)[0]
        if len(bioactive_conf_indexes) :
            ranker_results['bioactive_conf_first_index'] = int(bioactive_conf_indexes[0])
            ranker_results['bioactive_conf_first_normalized_index'] = int(bioactive_conf_indexes[0]) / n_poses
        
        return ranker_results
                
    def evaluate_poses(self,
                      mol, 
                      native_ligand):
        
        scores = []
        ligand_rmsds = []
        overlay_rmsds = []
        for pose in mol.GetConformers() :
            # rmsd = GetBestRMS(rdkit_mol, rdkit_mol, conf_id1, conf_id2)
            # mol.remove_atoms([atom for atom in mol.atoms if atom.atomic_number < 2])
            score = pose.GetProp('Gold.PLP.Fitness')
            if isinstance(score, str):
                score = score.strip()
                score = float(score)
            scores.append(score)
            ccdc_mol = self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=mol, conf_id=pose.GetId())
            ligand_rmsds.append(MolecularDescriptors.rmsd(native_ligand, 
                                                          ccdc_mol))
            overlay_rmsds.append(MolecularDescriptors.rmsd(native_ligand, 
                                                           ccdc_mol,
                                                           overlay=True))
        return scores, ligand_rmsds, overlay_rmsds
        
    
    def analysis_report(self, 
                        split='random',
                        split_i=0,
                        only_good_docking=True,
                        task: str = 'all',
                        results_dir: str = '/home/bb596/hdd/pdbbind_bioactive/results') :
        
        task_dir = os.path.join(results_dir, 
                                f'{split}_split_{split_i}/', 
                                task)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        
        # Load ranking results
        evaluation_name = f'{split}_split_{split_i}'
        evaluation_dir = os.path.join(results_dir, evaluation_name)
        ranker_name = 'bioschnet'
        ranker_dir = os.path.join(evaluation_dir, ranker_name)
        mol_results_path = os.path.join(ranker_dir, 'ranker_mol_results.p')
        with open(mol_results_path, 'rb') as f:
            all_mol_results = pickle.load(f)
            
        # Determine PDB ids to summarize
        if task in ['hard', 'easy']:
            pdb_ids = []
            for name, results in tqdm(all_mol_results.items()):
                if 'bioactive_like' in results:
                    n_confs = results['bioactive_like']['n_confs']
                    n_bio_like = results['bioactive_like']['n_masked']
                    ratio = n_bio_like / n_confs
                    hard_condition = (ratio < 0.05) and (n_confs == 250) and (task == 'hard') 
                    easy_condition = (ratio > 0.05) and (task == 'easy') 
                    if hard_condition or easy_condition:
                        pdb_id_subset = self.pdbbind_df[self.pdbbind_df['ligand_name'] == name]['pdb_id'].values
                        pdb_ids.extend(pdb_id_subset)
        else:
            pdb_ids = os.listdir(self.output_dir)
            
        results = []
        for pdb_id in pdb_ids : 
            result_path = os.path.join(self.output_dir, pdb_id, f'results_{split}_{split_i}.json')
            if os.path.exists(result_path) :
                with open(result_path, 'r') as f :
                    results.append(json.load(f))
        
        rankers = ['bioschnet', 'MMFF94s_energy', 'CCDC', 'shuffle', 'Gold.PLP.Fitness']
        metrics = ['score', 'ligand_rmsd', 'overlay_rmsd', 'first_successful_pose', 'correct_conf']
        top_indexes = {metric: defaultdict(list)
                       for metric in metrics}
        
        recalls = {}
        top_values = {}
        recall_metrics = ['top_score', 'top_rmsd']
        for m in recall_metrics:
            recalls[m] = {}
            top_values[m] = {}
            for r in rankers:
                recalls[m][r] = {}
                top_values[m][r] = {}
                for f in self.fractions:
                    recalls[m][r][f] = [] # list of bool values
                    top_values[m][r][f] = [] # list of float values
        
        flexible_successful_pose = defaultdict(list)
        flexible_generation_power = defaultdict(list)
        rigid_successful_pose = defaultdict(list)
        rigid_generation_power = defaultdict(list)
        
        for result in tqdm(results) :
            rigid_result = result['rigid']
            flexible_result = result['flexible']
            
            if 'overlay_rmsd_top_overlay_rmsd' in rigid_result :
                
                top_overlay_rmsd = rigid_result['overlay_rmsd_top_overlay_rmsd']
                has_bioactive_like = top_overlay_rmsd <= self.pose_rmsd_threshold
                has_250_generated = result['n_poses'] == 250
                task_condition = ((task == 'hard' and has_250_generated) 
                                  or (task == 'easy' and not has_250_generated) 
                                  or task == 'all')
                rigid_good_docking = rigid_result['ligand_rmsd_top_rmsd'] <= self.conf_rmsd_threshold
                # flexible_good_docking = flexible_result['top_ligand_rmsd'] <= self.bioactive_rmsd_threshold
                good_docking = rigid_good_docking # or flexible_good_docking
                good_docking_condition = ((only_good_docking and good_docking) or not only_good_docking)
                
                if has_bioactive_like and good_docking_condition and task_condition:
                
                    for r in rankers:
                        
                        for f in self.fractions:
                            f_str = str(f)
                            # import pdb;pdb.set_trace()
                            recalls['top_score'][r][f].append(rigid_result[r][f_str]['top_score_is_bioactive_pose'])
                            recalls['top_rmsd'][r][f].append(rigid_result[r][f_str]['top_rmsd_is_bioactive_pose'])
                            
                            top_values['top_score'][r][f].append(rigid_result[r][f_str]['top_score_rmsd'])
                            top_values['top_rmsd'][r][f].append(rigid_result[r][f_str]['top_rmsd_score'])
                        
                        top_indexes['score'][r].append(rigid_result[r]['1.0']['top_score_norm_index']) # 1 is the full fraction
                        top_indexes['ligand_rmsd'][r].append(rigid_result[r]['1.0']['top_rmsd_norm_index'])
                        top_indexes['overlay_rmsd'][r].append(rigid_result[r]['1.0']['top_overlay_rmsd_norm_index'])
                        if 'bioactive_pose_first_index' in rigid_result[r] :
                            top_indexes['first_successful_pose'][r].append(rigid_result[r]['bioactive_pose_first_index'])
                        if 'bioactive_conf_first_index' in rigid_result[r] :
                            top_indexes['correct_conf'][r].append(rigid_result[r]['bioactive_conf_first_index'])
            
                    # Successful pose and bioactive-like conf
                    # Flexible
                    top_score_successful_pose = flexible_result['ligand_rmsd_top_score'] <= self.pose_rmsd_threshold
                    flexible_successful_pose['top_score_pose'].append(top_score_successful_pose)
                    
                    top_rmsd_successful_pose = flexible_result['top_ligand_rmsd'] <= self.pose_rmsd_threshold
                    flexible_successful_pose['top_rmsd_pose'].append(top_rmsd_successful_pose)
                    
                    top_score_bioactive_conf = flexible_result['overlay_rmsd_top_score'] <= self.conf_rmsd_threshold
                    flexible_generation_power['top_score_conf'].append(top_score_bioactive_conf)
                    
                    # Rigid
                    top_score_successful_pose = rigid_result[r]['1.0']['top_score_is_bioactive_pose']
                    rigid_successful_pose['top_score_pose'].append(top_score_successful_pose)
                    
                    top_rmsd_successful_pose = rigid_result[r]['1.0']['top_rmsd_is_bioactive_pose']
                    rigid_successful_pose['top_rmsd_pose'].append(top_rmsd_successful_pose)
                    
                    top_score_bioactive_conf = rigid_result['overlay_rmsd_top_score'] <= self.conf_rmsd_threshold
                    rigid_generation_power['top_score_conf'].append(top_score_bioactive_conf)
        
        # Successful pose and bioactive-like conf
        # Flexible
        recall_flexible_top_score = np.array(flexible_successful_pose['top_score_pose'])
        recall_flexible_top_rmsd = np.array(flexible_successful_pose['top_rmsd_pose'])
        recall_flexible_overlay_rmsd = np.array(flexible_generation_power['top_score_conf'])
        n_flexible_mols = len(recall_flexible_top_score)
        
        flexible_results = {}
        
        print('Flexible n_mols')
        print(n_flexible_mols)
        
        flexible_results['n_mols'] = n_flexible_mols
        
        recall_flexible_top_score_fraction = recall_flexible_top_score.sum() / n_flexible_mols
        print('Flexible docking power top score')
        print(recall_flexible_top_score_fraction)
        flexible_results['top_score_recall'] = recall_flexible_top_score_fraction
        
        recall_flexible_top_rmsd_fraction = recall_flexible_top_rmsd.sum() / n_flexible_mols
        print('Flexible docking power top pose')
        print(recall_flexible_top_rmsd_fraction)
        flexible_results['top_rmsd_recall'] = recall_flexible_top_rmsd_fraction
        
        flexible_generation_power_fraction = recall_flexible_overlay_rmsd.sum() / n_flexible_mols
        print('Flexible generation power')
        print(flexible_generation_power_fraction)
        flexible_results['generation_power'] = flexible_generation_power_fraction
        
        flexible_results_path = os.path.join(task_dir,
                                             f'flexible_results.csv')
        with open(flexible_results_path, 'wb') as f:
            pickle.dump(flexible_results, f)
        
        recall_rigid_top_score = np.array(rigid_successful_pose['top_score_pose'])
        recall_rigid_top_rmsd = np.array(rigid_successful_pose['top_rmsd_pose'])
        recall_rigid_overlay_rmsd = np.array(rigid_generation_power['top_score_conf'])
        n_rigid_mols = len(recall_rigid_overlay_rmsd)
        
        # Rigid
        print('Rigid n_mols')
        print(n_rigid_mols)
        
        print('Rigid docking power top score')
        print(recall_rigid_top_score.sum() / n_rigid_mols)
        
        print('Rigid docking power top pose')
        print(recall_rigid_top_rmsd.sum() / n_rigid_mols)
        
        print('Rigid generation power')
        print(recall_rigid_overlay_rmsd.sum() / n_rigid_mols)
        
        if only_good_docking :
            suffix = 'successful_only'
        else :
            suffix = 'all'
        
        # Successful poses per fraction
        metric_names = {'top_score': 'Top Gold.PLP.Fitness',
                        'top_rmsd': 'Top RMSD'}
        rows = []
        for m in recall_metrics:
            for r in rankers:
                for f in self.fractions:
                    for v in recalls[m][r][f]:
                        row = {}
                        row['Metric'] = metric_names[m]
                        row['Ranker'] = r
                        row['Fraction'] = f
                        row['Value'] = int(v)
                        rows.append(row)
        results_df = pd.DataFrame(rows)
        grouped_df = results_df.groupby(['Metric', 'Ranker', 'Fraction'], sort=False).mean().reset_index()
            
        # save recalls
        df_path = os.path.join(task_dir,
                                f'rigid_ligand_docking_recall_{suffix}.csv')
        grouped_df.to_csv(df_path)
        
        for metric in recall_metrics:
            
            sns.lineplot(data=grouped_df[grouped_df['Metric'] == metric_names[metric]], 
                         x='Fraction', 
                         y='Value', 
                         hue='Ranker')
            
            if metric == 'top_rmsd':
                recall_flexible = recall_flexible_top_rmsd.sum()
            elif metric == 'top_score' :
                recall_flexible = recall_flexible_top_score.sum()
            plt.axhline(y=recall_flexible / n_flexible_mols, 
                            label='Flexible docking',
                            color='grey')
            
            plt.xlabel('Fraction of ranked conformations')
            plt.ylabel(metric_names[metric])
            plt.legend()
            plt.ylim(0, 1)
                
            fig_name = f'retrieval_{metric}_{split}_{suffix}.png'
            save_path = os.path.join(task_dir, 
                                     fig_name)
            plt.savefig(save_path)
            plt.clf()
        
        # Best pose index (useful to check on successful poses only)
        index_metrics = {'ligand_rmsd': 'best RMSD'}
        rows = []
        for m in index_metrics:
            for r in rankers:
                for v in top_indexes[m][r]:
                    row = {}
                    row['Metric'] = index_metrics[m]
                    row['Ranker'] = r
                    row['Value'] = v
                    rows.append(row)
        results_df = pd.DataFrame(rows)
        # save recalls
        df_path = os.path.join(task_dir,
                                f'rigid_ligand_docking_best_indexes_{suffix}.csv')
        results_df.to_csv(df_path)
        
        for metric in index_metrics:
            sns.ecdfplot(data=results_df[results_df['Metric'] == index_metrics[metric]], 
                        x='Value', 
                        hue='Ranker')
            plt.xlabel('Fraction of ranked conformations')
            plt.ylabel(f'Fraction of molecules with \n{index_metrics[metric]} pose')
            fig_name = f'retrieval_{metric}_{split}_{suffix}.png'
            save_path = os.path.join(task_dir, 
                                    fig_name)
            plt.savefig(save_path)
            plt.clf()
        
        
if __name__ == '__main__':

    rigid_docking = PDBbindDocking(use_cuda=False)
    
    splits = ['random', 'scaffold', 'protein']
    split_is = list(range(5))
    test_pdb_ids = rigid_docking.get_test_pdb_ids(splits=splits,
                                                  split_is=split_is)

    # rigid_docking.dock_molecule_pool(test_pdb_ids=test_pdb_ids)
    
    start_time = time.time()
    for split in splits :
        for split_i in split_is :
            print(split)
            print(split_i)
            # rigid_docking.docking_analysis_pool(split=split,
            #                                     split_i=split_i)#, single=True)
            for task in ['all', 'hard', 'easy']:
                print(task)
                print('All results')
                rigid_docking.analysis_report(split=split,
                                            split_i=split_i,
                                            task=task,
                                            only_good_docking=False)
                print('Only good docking results')
                rigid_docking.analysis_report(split=split,
                                            split_i=split_i,
                                            task=task,
                                            only_good_docking=True)
    runtime = time.time() - start_time
    print(f'{runtime} seconds runtime')
            