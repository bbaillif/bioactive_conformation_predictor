import os
from posixpath import split
import sys
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import desc
import torch
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from rdkit import Chem # must be called before ccdc import
from pdbbind_metadata_processor import PDBBindMetadataProcessor
from platinum_processor import PlatinumProcessor
from ccdc_rdkit_connector import CcdcRdkitConnector

from collections import defaultdict
from rdkit.ML.Scoring.Scoring import CalcEnrichment, CalcBEDROC
from ccdc.io import EntryReader
from molecule_featurizer import MoleculeFeaturizer
from litschnet import LitSchNet
from ccdc.docking import Docker
from ccdc.descriptors import MolecularDescriptors
from gold_docker import GOLDDocker
from pose_selector import (Pose, 
                           PoseSelectionError, 
                           RandomPoseSelector,
                           ScorePoseSelector,
                           EnergyPoseSelector,
                           ModelPoseSelector)
from multiprocessing import Pool


class RigidDocking() :
    
    def __init__(self,
                 dataset: str = 'pdbbind',
                 split_name: str = 'random',
                 split_i: int = 0,
                 output_dir_prefix: str='gold_docking',
                 use_cuda: bool=False,
                 bioactive_rmsd_threshold: float=1):
        
        assert dataset in ['pdbbind', 'platinum'], 'Dataset must be pdbbind or platinum'
        self.dataset = dataset
        self.split_name = split_name
        self.split_i = split_i
        self.output_dir = f'{output_dir_prefix}_{dataset}_{split_name}_{split_i}'
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
        
    def get_test_smiles(self) :
        if self.dataset == 'pdbbind' :
            path = os.path.join('data/',
                                f'{self.split_name}_splits',
                                f'test_smiles_{self.split_name}_split_{self.split_i}.txt')
            with open(path, 'r') as f:
                test_smiles = f.readlines()
                test_smiles = [smiles.strip() for smiles in test_smiles]
        elif self.dataset == 'platinum' :
            smiles_table = pd.read_csv('data/smiles_df.csv')
            test_smiles = smiles_table[(smiles_table['platinum']) 
                                       & (smiles_table['included'])]['smiles'].values
            
        return test_smiles
        
    def dock_molecule_conformations(self, 
                      ccdc_mols, # represents different conformations
                      pdb_id) :
        
        protein_path, ligand_path = self.data_processor.get_pdb_id_pathes(pdb_id=pdb_id)
        self.gold_docker = GOLDDocker(protein_path=protein_path,
                                      native_ligand_path=ligand_path,
                                      experiment_id=pdb_id,
                                      output_dir=self.output_dir,
                                      prepare_protein=self.prepare_protein)
        
        first_generated_mol = ccdc_mols[0]
        self.gold_docker.dock_molecule(ccdc_mol=first_generated_mol,
                                       mol_id=self.flexible_mol_id,
                                       n_poses=5)
        self.gold_docker.dock_molecules(ccdc_mols=ccdc_mols,
                                       mol_id=self.rigid_mol_id,
                                       n_poses=5,
                                       rigid=True)
        
    def dock_molecule_pool(self,
                           conf_ensembles):
        params = []
        for conf_ensemble in conf_ensembles :
            rdkit_mol = conf_ensemble.mol
            bioactive_conf_ids = [conf.GetId()
                                for conf in rdkit_mol.GetConformers()
                                if not conf.HasProp('Generator')]
            #import pdb; pdb.set_trace()
            generated_conf_ids = [conf.GetId()
                                for conf in rdkit_mol.GetConformers()
                                if conf.HasProp('Generator')]
            for conf_id in bioactive_conf_ids :
                bioactive_conf = rdkit_mol.GetConformer(conf_id)
                pdb_id = bioactive_conf.GetProp('PDB_ID')
                if pdb_id in self.data_processor.available_structures :
                    try :
                        if len(generated_conf_ids) == 100 :
                            params.append((rdkit_mol, pdb_id))
                    except :
                        print('Molecule pre-processing fails')
                    
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.dock_molecule_conformations_thread, params)
            
    def dock_molecule_conformations_thread(self,
                                           params) :
        rdkit_mol, pdb_id = params
        generated_conf_ids = [conf.GetId()
                                for conf in rdkit_mol.GetConformers()
                                if conf.HasProp('Generator')]
        ccdc_mols = [self.ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_mol,
                                                                        conf_id=conf_id)
                            for conf_id in generated_conf_ids]
        self.dock_molecule_conformations(ccdc_mols=ccdc_mols,
                                         pdb_id=pdb_id)
        
    def get_top_poses(self,
                      pdb_id,
                      rigid=True):
        """For multiple ligand docking, return the top pose for each ligand
        (a ligand is a distinct conf of a molecule in the context of rigid 
        docking)
        
        :param pdb_id: PDB_ID of the molecule, useful to retrieve output dir
        :type pdb_id: str
        :return: List of top poses, one per ligand name
        :rtupe: list[ccdc.docking.Docker.Results.DockedLigand]
        """
        
        top_poses = None 
        
        if rigid :
            subset = self.rigid_mol_id
        else :
            subset = self.flexible_mol_id
        docked_ligand_path = os.path.join(self.output_dir,
                                          pdb_id,
                                          subset,
                                          'docked_ligands.mol2')
        if os.path.exists(docked_ligand_path) :
            poses_reader = EntryReader(docked_ligand_path)
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
                          pdb_id,
                          conf_ensemble_library) :
        
        rdkit_native_ligand = [ce.mol 
                        for smiles, ce in conf_ensemble_library.get_unique_molecules()
                        if ce.mol.GetConformer().GetProp('PDB_ID') == pdb_id][0]
        
        return rdkit_native_ligand
    
    def analyze_pdb_id(self,
                       pdb_id,
                       rdkit_native_ligand) :
        
        ccdc_rdkit_connector = CcdcRdkitConnector()
        native_ligand = ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_native_ligand,
                                                                    conf_id=0)
        
        print(f'Analyzing {pdb_id}')
        results = {}
        results['flexible'] = {}
        results['rigid'] = {}
        flexible_poses = self.get_top_poses(pdb_id, rigid=False)
        rigid_poses = self.get_top_poses(pdb_id)
        if flexible_poses and rigid_poses :
            # protein_path, ligand_path = self.pdbbind_metadata_processor.get_pdb_id_pathes(pdb_id=pdb_id)
            # native_ligand = MoleculeReader(ligand_path)[0]
            # previous lines not working because not the same atom ordering
            
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

                print(f'Save {pdb_id}')
                results_path = os.path.join(self.output_dir, pdb_id, 'results.json')
                with open(results_path, 'w') as f :
                    json.dump(results, f)
                    
        else :
            print(f'No pose docked for {pdb_id}')
    
    
    def docking_analysis_thread(self, 
                                params):
        pdb_id, rdkit_native_ligand = params
        self.analyze_pdb_id(pdb_id, rdkit_native_ligand)
    
    
    def prepare_analysis(self) :
        
        self.mol_featurizer = MoleculeFeaturizer()
        
        self.model_checkpoint_dir = os.path.join('lightning_logs',
                                                  f'{self.split_name}_split_{self.split_i}_new',
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
    
    def docking_analysis_pool(self):
        
        self.prepare_analysis()
        
        with open('data/raw/ccdc_generated_conf_ensemble_library.p', 'rb') as f:
            conf_ensemble_library = pickle.load(f)
            
        params = []
        pdb_ids = os.listdir(self.output_dir)
        for pdb_id in pdb_ids :
            rdkit_native_ligand = self.get_native_ligand(pdb_id, 
                                                   conf_ensemble_library=conf_ensemble_library)
            params.append((pdb_id, rdkit_native_ligand))
            
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.docking_analysis_thread, params)
    
                
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
            mol.remove_atoms([atom for atom in mol.atoms if atom.atomic_number < 2])
            scores.append(pose.fitness())
            ligand_rmsds.append(MolecularDescriptors.rmsd(native_ligand, 
                                                          mol))
            overlay_rmsds.append(MolecularDescriptors.rmsd(native_ligand, 
                                                           mol,
                                                           overlay=True))
        return scores, ligand_rmsds, overlay_rmsds
        
    
    def analysis_report(self) :
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
        
        rankers = ['model', 'energy', 'score', 'random', 'CCDC']
        for result in results :
            rigid_result = result['rigid']
            if rigid_result['top_rmsd'] <= self.bioactive_rmsd_threshold :
                for ranker in rankers :
                    top_indexes['score'][ranker].append(rigid_result[ranker]['top_score_index'])
                    top_indexes['ligand_rmsd'][ranker].append(rigid_result[ranker]['top_rmsd_index'])
                    top_indexes['overlay_rmsd'][ranker].append(rigid_result[ranker]['top_overlay_rmsd_index'])
                    top_indexes['docking_power'][ranker].append(rigid_result[ranker]['bioactive_pose_first_index'])
                    top_indexes['correct_conf'][ranker].append(rigid_result[ranker]['bioactive_conf_first_index'])
        
        # Produce lineplots
        for metric, top_indexes_metric in top_indexes.items():
            for ranker, top_indexes_task in top_indexes_metric.items():
                
                top_indexes_task = np.array(top_indexes_task)
            
                thresholds = range(100)
                recalls = []
                for threshold in thresholds :
                    recalls.append(np.sum(top_indexes_task <= threshold))
                    
                sns.lineplot(x=thresholds, y=recalls, label=ranker)
            
            #plt.axhline(y=flexible_score, label='flexible')
                
            plt.title(f'Retrieval of top {metric}')
            plt.xlabel('Conformation rank')
            plt.ylabel('Number of retrieved molecule (rank is best)')
            plt.savefig(f'retrieval_{metric}')
            plt.clf()
        
        
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
        
    with open('data/raw/ccdc_generated_conf_ensemble_library.p', 'rb') as f:
        conf_ensemble_library = pickle.load(f)
    
    test_smiles = rigid_docking.get_test_smiles()
    
    conf_ensembles = []
    for smiles in test_smiles :
        try :
            conf_ensemble = conf_ensemble_library.get_conf_ensemble(smiles=smiles)
            conf_ensembles.append(conf_ensemble)
        except KeyError:
            print('smiles not found in conf_ensemble')
    
    rigid_docking.dock_molecule_pool(conf_ensembles=conf_ensembles)
            