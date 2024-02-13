import os
import numpy as np
import pandas as pd
import json
import time
import copy
import pickle
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem # must be called before ccdc import
from rdkit.Chem import Mol
from rdkit.Chem.rdMolAlign import GetBestRMS, CalcRMS
from tqdm import tqdm
from collections import defaultdict
from typing import List, Sequence, Dict, Any, Tuple
from ccdc.descriptors import MolecularDescriptors
from ccdc.molecule import Molecule
from multiprocessing import Pool, TimeoutError

from .gold_docker import GOLDDocker
from bioconfpred.data.utils import MolConverter
from bioconfpred.data.sources import PDBbind
from bioconfpred.conf_ensemble import ConfEnsemble
from bioconfpred.ranker import ConfRanker
from bioconfpred.data.split import DataSplit
from bioconfpred.params import (GOLD_PDBBIND_DIRPATH,
                    DATA_DIRPATH, 
                    PDBBIND_DIRPATH,
                    BIO_CONF_DIRNAME,
                    BIO_CONF_DIRPATH,
                    GEN_CONF_DIRNAME,
                    RESULTS_DIRPATH)

if not os.path.exists('logs/'):
    os.mkdir('logs/')
logging.basicConfig(filename='logs/pdbbind_docking.log')

class PDBbindDocking() :
    """Performs batch re-docking of PDBbind ligands

    :param output_dir: Directory where dockign results are stored,
        defaults to GOLD_PDBBIND_DIRPATH
    :type output_dir: str, optional
    :param data_dir: Directory where the bioactive+generated conformatiosn are stored,
        defaults to DATA_DIRPATH
    :type data_dir: str, optional
    :param pose_rmsd_threshold: RMSD threshold to define the successful docking,
        defaults to 2
    :type pose_rmsd_threshold: float, optional
    :param conf_rmsd_threshold: Overlay RMSD threshold to define the 
        bioactive-like conformer, defaults to 1
    :type conf_rmsd_threshold: float, optional
    :param rmsd_backend: RMSD computation method, rdkit or ccdc, 
        defaults to 'ccdc'
    :type rmsd_backend: str, optional
    :param max_confs: Maximum number of docked conformers, defaults to 250
    :type max_confs: int, optional
    """
    
    def __init__(self,
                 output_dir: str = GOLD_PDBBIND_DIRPATH,
                 data_dir: str = DATA_DIRPATH,
                 pose_rmsd_threshold: float = 2,
                 conf_rmsd_threshold: float = 1,
                 rmsd_backend: str = 'ccdc',
                 max_confs: int = 250):
        

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir) :
            os.mkdir(self.output_dir)
        self.data_dir = data_dir
        self.pose_rmsd_threshold = pose_rmsd_threshold
        self.conf_rmsd_threshold = conf_rmsd_threshold
        
        assert rmsd_backend in ['ccdc', 'rdkit'], \
            'Backend for RMSD calculation must be ccdc or rdkit'
        self.rmsd_backend = rmsd_backend
        
        self.max_confs = max_confs
        
        self.mol_converter = MolConverter()
        
        self.flexible_mol_id = 'flexible_mol'
        self.rigid_mol_id = 'rigid_confs'
        
        self.pdbbind = PDBbind(root=PDBBIND_DIRPATH)
        self.prepare_protein = False
            
        self.pdbbind_df = pd.read_csv(os.path.join(BIO_CONF_DIRPATH, 'pdb_df.csv'))
        self.cel_df = pd.read_csv(os.path.join(BIO_CONF_DIRPATH, 'ensemble_names.csv'))
        
        self.fractions = np.around(np.arange(0.01, 1.01, 0.01), 2)
    
        
    def dock_mol_confs(self, 
                        ccdc_mols: List[Molecule], # represents different conformations
                        pdb_id: str,
                        overwrite: bool = False
                        ) -> Dict[str, Any]:
        """Dock the given molecules in the protein given by pdb_id

        :param ccdc_mols: Input molecules 
        :type ccdc_mols: List[Molecule]
        :param pdb_id: PDB ID to retrieve the protein
        :type pdb_id: str
        :param overwrite: Set to True to erase previous results for the pdb_id,
            defaults to False
        :type overwrite: bool, optional
        :return: Dictionary of results for rigid and flexible ligand docking
        :rtype: Dict[str, Any]
        """
        
        protein_path, ligand_pathes = self.pdbbind.get_pdb_id_pathes(pdb_id=pdb_id)
        
        ccdc_mols = ccdc_mols[:self.max_confs]
        
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
                                                        return_runtime=True,
                                                        overwrite=overwrite)
            results['flexible']['runtime'] = runtime
            
            # Rigid
            _, runtime = self.gold_docker.dock_molecules(ccdc_mols=ccdc_mols,
                                                        mol_id=self.rigid_mol_id,
                                                        n_poses=10,
                                                        rigid=True,
                                                        return_runtime=True,
                                                        overwrite=overwrite)
            results['rigid']['runtime'] = runtime
            results_path = os.path.join(self.output_dir, experiment_id, 'results.json')
            with open(results_path, 'w') as f :
                json.dump(results, f)
                
        return results
        
        
    def get_ce_from_pdb_id(self,
                           pdb_id: str,
                           library_name: str = GEN_CONF_DIRNAME,
                           ) -> ConfEnsemble:
        """Get the conf ensemble from a pdb_id. Allows to retrieve
        the generated conformers for a ligand when stored in a conf
        ensemble library

        :param pdb_id: PDB ID to retrieve the ligand
        :type pdb_id: str
        :param library_name: Name of the library, defaults to GEN_CONF_DIRNAME
        :type library_name: str, optional
        :return: Ensemble of generated conformers for the ligand
        :rtype: ConfEnsemble
        """
        name = self.pdbbind_df[self.pdbbind_df['pdb_id'] == pdb_id]['ligand_name'].values[0]
        filename = self.cel_df[self.cel_df['ensemble_name'] == name]['filename'].values[0]
        filepath = os.path.join(self.data_dir, library_name, filename)
        ce = ConfEnsemble.from_file(filepath, name=name)
        return ce
        
        
    def dock_mol_pool(self,
                      test_pdb_ids: List[str]) -> None:
        """Dock all ligands given in the test_pdb_ids

        :param test_pdb_ids: List of PDB IDs to re-dock
        :type test_pdb_ids: List[str]
        """
        params = []
        
        assert len(test_pdb_ids) > 0, 'you must provide test pdb ids'
        
        logging.info('Prepare conformations for docking')
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
                    logging.warning(f'{pdb_id} not included')
                    logging.warning(str(e))
                    
        logging.info(f'Number of threads : {len(params)}')
        with Pool(processes=20, maxtasksperchild=1) as pool :
            # pool.map(self.dock_mol_confs_thread, params)
            iterator = pool.imap(self.dock_mol_confs_thread, params)
            done_looping = False
            while not done_looping:
                try:
                    try:
                        item = iterator.next(timeout=600)
                    except TimeoutError:
                        logging.warning("Docking is too long, returning TimeoutError")

                except StopIteration:
                    done_looping = True
            
            
    def dock_mol_confs_thread(self,
                                params: Tuple[Mol, str]
                                ) -> Dict[str, Any]:
        rdkit_mol, pdb_id = params
        ccdc_mols = [self.mol_converter.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_mol,
                                                                conf_id=conf_id)
                    for conf_id in range(rdkit_mol.GetNumConformers())]
        results = None
        try :
            results = self.dock_mol_confs(ccdc_mols=ccdc_mols,
                                          pdb_id=pdb_id)
        except Exception as e :
            logging.warning(e)
            
        return results
        
        
    def get_top_poses_rigid(self,
                            pdb_id: str) -> Mol:
        """For multiple ligand docking, return the top pose for each ligand
        (a ligand is a distinct conf of a molecule in the context of rigid 
        docking)
        
        :param pdb_id: PDB_ID of the molecule, useful to retrieve output dir
        :type pdb_id: str
        :return: RDKit molecule containing the top scoring docking pose of 
            each (generated) conformer
        :rtype: Mol
        """
 
        docked_ligand_path = os.path.join(self.output_dir,
                                          pdb_id,
                                          self.rigid_mol_id,
                                          'docked_ligands.mol2')
        if os.path.exists(docked_ligand_path) :
            ce = ConfEnsemble.from_file(docked_ligand_path,
                                        docked_poses=True)
            new_mol = Mol(ce.mol)
            new_mol.RemoveAllConformers()

            seen_ligands = []
            for pose in ce.mol.GetConformers() :
                identifier = pose.GetProp('_Name')
                lig_num = identifier.split('|')[1]
                if not lig_num in seen_ligands :
                    seen_ligands.append(lig_num)
                    new_mol.AddConformer(pose)
                    
        return new_mol
    
    def get_native_ligand(self,
                          pdb_id: str) -> Mol:
        """Retrieves the native ligand given a PDB ID

        :param pdb_id: Input PDB ID
        :type pdb_id: str
        :return: RDKit molecule of the native ligand
        :rtype: Mol
        """
        
        conf_prop = 'PDB_ID'
        
        ce = self.get_ce_from_pdb_id(pdb_id, 
                                    library_name=BIO_CONF_DIRNAME)
        
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
            logging.warning(f'{pdb_id} not found')
        
        return rdkit_native_ligand
    
    
    def docking_analysis_pool(self,
                              rankers: List[ConfRanker],
                              pdb_ids: Sequence[str],
                              data_split: DataSplit,
                              single_thread: bool = False
                              ) -> None:
        """Analyze results from PDBbind re-docking

        :param rankers: List of conformer rankers
        :type rankers: List[ConfRanker]
        :param pdb_ids: List of re-docked PDB IDs to compile
        :type pdb_ids: Sequence[str]
        :param data_split: Data split for corresponding analysis (i.e. random, 
            scaffold)
        :type data_split: DataSplit
        :param single_thread: Go one by one on the threads instead of using pool,
            defaults to False
        :type single_thread: bool, optional

        """
        
        self.rankers = rankers
        self.data_split = data_split
            
        params = []
        for pdb_id in tqdm(pdb_ids) :
            try :
                native_ligand = self.get_native_ligand(pdb_id)
                # if native_ligand.GetNumHeavyAtoms() <= 50:
                params.append((pdb_id, native_ligand))
                # else:
                #     logging.info(f'{pdb_id} ligand is large (> 50 HA), not included')
            except Exception as e :
                logging.warning(f'{pdb_id} failed')
                logging.warning(str(e))
            
        logging.info(f'Number of threads : {len(params)}')
        if single_thread :
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
                            logging.warning("Analysis is too long, returning TimeoutError")
                            return 0
                    except StopIteration:
                        done_looping = True
    
    
    def docking_analysis_thread(self, 
                                params: Tuple[str, Mol]
                                ) -> Dict[str, Any]:
        """Single thread of a docking analysis (for one ligand)

        :param params: PDB ID and rdkit mol of the native ligand
        :type params: Tuple[str, Mol]
        :return: Dictionnary of results
        :rtype: Dict[str, Any]
        """
        
        pdb_id, rdkit_native_ligand = params
        results = None
        try :
            results = self.analyze_pdb_id(pdb_id, rdkit_native_ligand)
        except Exception as e :
            logging.warning(f'Evaluation failed for {pdb_id}')
            logging.warning(str(e))
            
        return results
    
    
    def analyze_pdb_id(self,
                       pdb_id: str,
                       native_ligand: Mol) -> Dict[str, Any]:
        """Analyse the re-docking results for a ligand

        :param pdb_id: Input PDB ID
        :type pdb_id: str
        :param native_ligand: RDKit molecule of the native ligand
        :type native_ligand: Mol
        :return: Dictionnary of results
        :rtype: Dict[str, Any]
        """
        
        split_type = self.data_split.split_type
        split_i = self.data_split.split_i
        
        logging.info(f'Analyzing {pdb_id}')
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
        logging.info(f'Loading flexible docked ligands')
        if os.path.exists(docked_ligand_path) :
            ce_flexible_poses = ConfEnsemble.from_file(docked_ligand_path, 
                                                       docked_poses=True)
            flexible_poses_mol = ce_flexible_poses.mol
        
        logging.info(f'Loading rigid docked ligands')
        rigid_poses_mol = self.get_top_poses_rigid(pdb_id)
        results['n_poses'] = rigid_poses_mol.GetNumConformers()
        
        scores, ligand_rmsds, overlay_rmsds = self.compute_rmsds_scores(rigid_poses_mol, 
                                                                   native_ligand)
        
        if flexible_poses_mol and rigid_poses_mol :
            
            included = True
            rigid_results = {}
            for ranker in self.rankers:
                logging.info(f'Using {ranker.name}')
                try :
                    logging.info(f'Ranking conformations')
                    ranks = ranker.rank_molecule(rigid_poses_mol)
                except Exception as e:
                    logging.warning(ranker.name)
                    logging.warning(str(e))
                    logging.warning(f'No pose selected for {pdb_id}')
                    included = False
                    break
                else:
                    logging.info(f'Evaluate ranked conformations')
                    ranker_results = self.evaluate_ranker(ranks,
                                                          scores,
                                                          ligand_rmsds,
                                                          overlay_rmsds)
                    rigid_results[ranker.name] = ranker_results
                    ranker_results_filepath = f'results_{split_type}_{split_i}_{ranker.name}'
                    ranker_split_results_path = results_path.replace('results', 
                                                                      ranker_results_filepath)
                    with open(ranker_split_results_path, 'w') as f :
                        json.dump(ranker_results, f)
                    
            if included :
                
                for ranker in self.rankers:
                    results['rigid'][ranker.name] = rigid_results[ranker.name]
                
                # Score/RMSD figures for rigid poses
                # scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(rigid_poses_mol,
                #                                                           native_ligand)
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
                scores, ligand_rmsds, overlay_rmsds = self.compute_rmsds_scores(ce_flexible_poses.mol,
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

                logging.info(f'Save {pdb_id}')
                results_filepath = f'results_{split_type}_{split_i}'
                split_results_path = results_path.replace('results', 
                                                          results_filepath)
                with open(split_results_path, 'w') as f :
                    json.dump(results, f)
                    
            else :
                logging.warning(f'{pdb_id} not included')
                    
        else :
            logging.warning(f'No pose docked for {pdb_id}')
            
        return results
    
                
    def evaluate_ranker(self, 
                        ranks: np.ndarray,
                        scores: np.ndarray,
                        ligand_rmsds: np.ndarray,
                        overlay_rmsds: np.ndarray
                        ) -> Dict[str, Any]:
        """Evaluate the early enrichment of bioactive-like conformer/poses from 
        a list of conformers based on the given ranks from a ranker

        :param ranks: Rank for each conformer
        :type ranks: np.ndarray
        :param scores: Docking score for each conformer
        :type scores: np.ndarray
        :param ligand_rmsds: RMSD to bioactive pose for each conformer
        :type ligand_rmsds: np.ndarray
        :param overlay_rmsds: Overlay RMSD (to bioactive conf) for each conformer
        :type overlay_rmsds: np.ndarray
        :return: Dictionnary of results
        :rtype: Dict[str, Any]
        """
        ranker_results = {}
        sorted_indexes = np.argsort(ranks)
        scores = scores[sorted_indexes]
        ligand_rmsds = ligand_rmsds[sorted_indexes]
        overlay_rmsds = overlay_rmsds[sorted_indexes]
        
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
                
    def compute_rmsds_scores(self,
                            mol: Mol, 
                            native_ligand: Mol
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the docking scores and Pose RMSD + Overlay RMSD for the ligand

        :param mol: RDKit mol containing conformer poses
        :type mol: Mol
        :param native_ligand: RDKit mol of the native pose
        :type native_ligand: Mol
        :return: Scores, pose RMSD and overlay RMSD for each conformer
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        
        mol_noH = Chem.RemoveHs(mol)
        native_noH = Chem.RemoveHs(native_ligand)
        
        scores = []
        ligand_rmsds = []
        overlay_rmsds = []
        logging.info(f'Computing ARMSD')
        for pose in mol.GetConformers() :
            score = pose.GetProp('Gold.PLP.Fitness')
            if isinstance(score, str):
                score = score.strip()
                score = float(score)
            scores.append(score)
            
            if self.rmsd_backend == 'rdkit':
                rmsd = CalcRMS(mol_noH, native_noH, pose.GetId(), 0)
                overlay_rmsd = GetBestRMS(mol_noH, native_noH, pose.GetId(), 0, 
                                        maxMatches=100)
                
            elif self.rmsd_backend == 'ccdc':
                ccdc_mol = self.mol_converter.rdkit_conf_to_ccdc_mol(rdkit_mol=mol, 
                                                                            conf_id=pose.GetId())
                ccdc_native_ligand = self.mol_converter.rdkit_conf_to_ccdc_mol(rdkit_mol=native_ligand)
                rmsd = MolecularDescriptors.rmsd(ccdc_mol, 
                                                 ccdc_native_ligand)
                overlay_rmsd = MolecularDescriptors.rmsd(ccdc_mol, 
                                                         ccdc_native_ligand,
                                                         overlay=True)
                
            ligand_rmsds.append(rmsd)
            overlay_rmsds.append(overlay_rmsd)
            
        scores = np.array(scores)
        ligand_rmsds = np.array(ligand_rmsds)
        overlay_rmsds = np.array(ligand_rmsds)
        
        return scores, ligand_rmsds, overlay_rmsds
        
    
    def analysis_report(self, 
                        data_split: DataSplit,
                        rankers: List[ConfRanker],
                        task: str = 'all',
                        only_good_docking: bool = True,
                        results_dir: str = RESULTS_DIRPATH) :
        """Produce summaries of analysis for a given test subset of a data split

        :param data_split: Data split to get the test subset
        :type data_split: DataSplit
        :param rankers: Conf rankers to test
        :type rankers: List[ConfRanker]
        :param task: all, easy or hard, defaults to 'all'. Hards are all 
        molecules with a ratio of bioactive-like conformers lower than 0.05 and 
        250 generated conformers; easy are all molecules with a ratio of 
        bioactive-like conformers higher than 0.05
        :type task: str, optional
        :param only_good_docking: Only computes results for ligands where 
            re-docking retrieved a bioactive-like poses, defaults to True
        :type only_good_docking: bool, optional
        :param results_dir: Directory where to compile results, 
            defaults to RESULTS_DIRPATH
        :type results_dir: str, optional
        """
        
        split_type = data_split.split_type
        split_i = data_split.split_i
        
        task_dir = os.path.join(results_dir, 
                                f'{split_type}_split_{split_i}/', 
                                task)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        
        # Load ranking results
        evaluation_name = f'{split_type}_split_{split_i}'
        evaluation_dir = os.path.join(results_dir, evaluation_name)
        
        ranker_name = 'Shuffle'
        ranker_dir = os.path.join(evaluation_dir, ranker_name)
        mol_results_path = os.path.join(ranker_dir, 'ranker_mol_results.p')
        with open(mol_results_path, 'rb') as f:
            all_mol_results: dict = pickle.load(f)
            
        # Determine PDB ids to summarize
        pdb_ids = []
        for pdb_id, results in tqdm(all_mol_results.items()):
            if 'bioactive_like' in results:
                n_confs = results['bioactive_like']['n_confs']
                n_bio_like = results['bioactive_like']['n_masked']
                ratio = n_bio_like / n_confs
                hard_condition = (ratio < 0.05) and (n_confs == 250) and (task == 'hard') 
                easy_condition = (ratio > 0.05) and (task == 'easy') 
                if hard_condition or easy_condition or task == 'all' :
                    # pdb_id_subset = self.pdbbind_df[self.pdbbind_df['ligand_name'] == name]['pdb_id'].values
                    # pdb_ids.extend(pdb_id_subset)
                    pdb_ids.append(pdb_id)
            
        logging.info('Reading files')
        results = []
        for pdb_id in tqdm(pdb_ids) : 
            result_path = os.path.join(self.output_dir, 
                                       pdb_id, 
                                       f'results_{split_type}_{split_i}.json')
            if os.path.exists(result_path) :
                with open(result_path, 'r') as f :
                    results.append(json.load(f))
        
        ranker_names = [ranker.name for ranker in rankers]
        metrics = ['score', 'ligand_rmsd', 'overlay_rmsd', 'first_successful_pose', 'correct_conf']
        top_indexes = {metric: defaultdict(list)
                       for metric in metrics}
        
        # Initialize variables to store results
        recalls = {}
        top_values = {}
        recall_metrics = ['top_score', 'top_rmsd']
        for m in recall_metrics:
            recalls[m] = {}
            top_values[m] = {}
            for r in ranker_names:
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
            
            # if task == 'hard':
            #     import pdb;pdb.set_trace()
            
            # import pdb;pdb.set_trace()
            
            # we might take pdb_id that were not tested with the latest rankers
            if all([r in rigid_result for r in ranker_names]):
            
                if 'overlay_rmsd_top_overlay_rmsd' in rigid_result :
                    
                    has_250_generated = result['n_poses'] == 250
                    task_condition = ((task == 'hard' and has_250_generated) 
                                    or (task == 'easy' and not has_250_generated) 
                                    or task == 'all')
                    rigid_good_docking = rigid_result['ligand_rmsd_top_rmsd'] <= self.pose_rmsd_threshold
                    # flexible_good_docking = flexible_result['top_ligand_rmsd'] <= self.bioactive_rmsd_threshold
                    good_docking = rigid_good_docking # or flexible_good_docking
                    good_docking_condition = ((only_good_docking and good_docking) or not only_good_docking)
                    
                    if good_docking_condition and task_condition:
                    
                        for r in ranker_names:
                                            
                            for f in self.fractions:
                                f_str = str(f)
                                recalls['top_score'][r][f].append(rigid_result[r][f_str]['top_score_is_bioactive_pose'])
                                recalls['top_rmsd'][r][f].append(rigid_result[r][f_str]['top_rmsd_is_bioactive_pose'])
                                
                                top_values['top_score'][r][f].append(rigid_result[r][f_str]['top_score_rmsd'])
                                top_values['top_rmsd'][r][f].append(rigid_result[r][f_str]['top_rmsd_score'])
                            
                            top_indexes['score'][r].append(rigid_result[r]['1.0']['top_score_norm_index']) # 1 is the full fraction
                            top_indexes['ligand_rmsd'][r].append(rigid_result[r]['1.0']['top_rmsd_norm_index'])
                            top_indexes['overlay_rmsd'][r].append(rigid_result[r]['1.0']['top_overlay_rmsd_norm_index'])
                            if 'bioactive_pose_first_index' in rigid_result[r] :
                                top_indexes['first_successful_pose'][r].append(rigid_result[r]['bioactive_pose_first_normalized_index'])
                            if 'bioactive_conf_first_index' in rigid_result[r] :
                                top_indexes['correct_conf'][r].append(rigid_result[r]['bioactive_conf_first_normalized_index'])
                
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
            for r in ranker_names:
                for f in self.fractions:
                    for v in recalls[m][r][f]:
                        row = {}
                        row['Metric'] = metric_names[m]
                        row['Ranker'] = r
                        row['Fraction'] = f
                        row['Value'] = int(v)
                        rows.append(row)
        results_df = pd.DataFrame(rows)
        try:
            grouped_df = results_df.groupby(['Metric', 'Ranker', 'Fraction'], sort=False).mean().reset_index()
        except Exception as e:
            logging.warning(str(e))
            
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
                
            fig_name = f'retrieval_{metric}_{split_type}_{suffix}.png'
            save_path = os.path.join(task_dir, 
                                     fig_name)
            plt.savefig(save_path)
            plt.clf()
        
        # Best pose index (useful to check on successful poses only)
        index_metrics = {'ligand_rmsd': 'best RMSD',
                         'first_successful_pose': 'first successful pose'}
        rows = []
        for m in index_metrics:
            for r in ranker_names:
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
            fig_name = f'retrieval_{metric}_{split_type}_{suffix}.png'
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

    # rigid_docking.dock_mol_pool(test_pdb_ids=test_pdb_ids)
    
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
            