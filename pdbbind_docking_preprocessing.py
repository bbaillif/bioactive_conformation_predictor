import os
import sys
import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from rdkit import Chem
from pdbbind_metadata_processor import PDBBindMetadataProcessor
from ccdc_rdkit_connector import CcdcRdkitConnector

from collections import defaultdict
from rdkit.ML.Scoring.Scoring import CalcEnrichment, CalcBEDROC
from ccdc.io import MoleculeReader
from molecule_featurizer import MoleculeFeaturizer
from litschnet import LitSchNet
from ccdc.docking import Docker
from ccdc.descriptors import MolecularDescriptors
from gold_docker import GOLDDocker
from pose_selector import (RandomPoseSelector,
                           ScorePoseSelector,
                           EnergyPoseSelector,
                           ModelPoseSelector)
from multiprocessing import Pool


class PDBBindDocking() :
    
    def __init__(self,
                 output_dir='gold_docking_pdbbind',
                 use_cuda=False):
        
        self.output_dir = output_dir
        self.use_cuda = use_cuda
        
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
        
        self.flexible_mol_id = 'flexible_mol'
        self.rigid_mol_id = 'rigid_confs'
        
    def dock_molecule_conformations(self, 
                      ccdc_mols, # represents different conformations
                      pdb_id) :
        
        self.pdbbind_metadata_processor = PDBBindMetadataProcessor()
        
        self.mol_featurizer = MoleculeFeaturizer()
        
        self.model_checkpoint_dir = os.path.join('lightning_logs',
                                                  'random_split_0_new',
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
        
        protein_path, ligand_path = self.pdbbind_metadata_processor.get_pdb_id_pathes(pdb_id=pdb_id)
        self.gold_docker = GOLDDocker(protein_path=protein_path,
                                      native_ligand_path=ligand_path,
                                      experiment_id=pdb_id,
                                      output_dir=self.output_dir)
        
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
            bioactive_conf = rdkit_mol.GetConformer(0)
            pdb_id = bioactive_conf.GetProp('PDB_ID')
            generated_conf_ids = [conf.GetId()
                                for conf in rdkit_mol.GetConformers()
                                if conf.HasProp('Generator')]
                
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
        docked_ligand_file = os.path.join(self.output_dir,
                                          pdb_id,
                                          subset,
                                          'docked_ligands.mol2')
        if os.path.exists(docked_ligand_file) :
            poses = Docker.Results.DockedLigandReader(docked_ligand_file, 
                                                        settings=None)

            top_poses = []
            seen_ligands = []
            for pose in poses :
                identifier = pose.identifier
                lig_num = identifier.split('|')[1]
                if not lig_num in seen_ligands :
                    top_poses.append(pose)
                    seen_ligands.append(lig_num)
                    
        return top_poses
    
    def docking_analysis(self,
                         pdb_ids):
        
        self.top_indexes = {}
        self.top_indexes['score'] = defaultdict(list)
        self.top_indexes['rmsd'] = defaultdict(list)
        
        flexible_result = defaultdict(list)
        
        with open('data/raw/ccdc_generated_conf_ensemble_library.p', 'rb') as f:
            conf_ensemble_library = pickle.load(f)
        
        for pdb_id in tqdm(pdb_ids) :
            flexible_poses = self.get_top_poses(pdb_id, rigid=False)
            rigid_poses = self.get_top_poses(pdb_id)
            if flexible_poses and rigid_poses :
                flexible_top_pose = flexible_poses[0]
                
                # protein_path, ligand_path = self.pdbbind_metadata_processor.get_pdb_id_pathes(pdb_id=pdb_id)
                # native_ligand = MoleculeReader(ligand_path)[0]
                
                ccdc_rdkit_connector = CcdcRdkitConnector()
                native_ligand = [ce.mol 
                                for smiles, ce in conf_ensemble_library.get_unique_molecules()
                                if ce.mol.GetConformer().GetProp('PDB_ID') == pdb_id][0]
                native_ligand = ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=native_ligand,
                                                                            conf_id=0)
                included = True
                for selector_name, pose_selector in self.pose_selectors.items() :
                    sorted_poses = pose_selector.select_poses(poses=rigid_poses)
                    
                    if sorted_poses and included: # if mol_featurizer fails, sorted_poses is None
                    
                        self.evaluate_ranker(poses=sorted_poses,
                                             native_ligand=native_ligand,
                                             ranker_name=selector_name)
                        
                    else :
                        included = False
                
                if included :
                    self.evaluate_ranker(poses=rigid_poses,
                                         native_ligand=native_ligand,
                                         ranker_name='CCDC')
                    
                    scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(poses=flexible_poses,
                                                                    native_ligand=native_ligand)
                    flexible_result['score'].append(scores[0])
                    flexible_result['rmsd'].append(ligand_rmsds[0])
            
        # Produce lineplots
        for metric, top_indexes_metric in self.top_indexes.items():
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
                
    def evaluate_ranker(self, 
                        poses, 
                        native_ligand,
                        ranker_name):
        scores, ligand_rmsds, overlay_rmsds = self.evaluate_poses(poses=poses,
                                                                  native_ligand=native_ligand)
        score_argsort = np.negative(scores).argsort()
        rmsd_argsort = np.array(ligand_rmsds).argsort()
        self.top_indexes['score'][ranker_name].append(score_argsort[0])
        self.top_indexes['rmsd'][ranker_name].append(rmsd_argsort[0])
        
                
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
                                                          mol, 
                                                          overlay=False))
            overlay_rmsds.append(MolecularDescriptors.rmsd(native_ligand, 
                                                           mol))
        return scores, ligand_rmsds, overlay_rmsds
        
        
if __name__ == '__main__':
    pdbbind_docking = PDBBindDocking()
    
    with open('data/random_splits/test_smiles_random_split_0.txt') as f:
        test_smiles = f.readlines()
        test_smiles = [smiles.strip() for smiles in test_smiles]
        
    with open('data/raw/ccdc_generated_conf_ensemble_library.p', 'rb') as f:
        conf_ensemble_library = pickle.load(f)
    
    conf_ensembles = []
    for smiles in test_smiles :
        try :
            conf_ensemble = conf_ensemble_library.get_conf_ensemble(smiles=smiles)
            conf_ensembles.append(conf_ensemble)
        except KeyError:
            print('smiles not found in conf_ensemble')
    
    pdbbind_docking.dock_molecule_pool(conf_ensembles=conf_ensembles)
    
    
    # ces = []
    # for smiles, ce in conf_ensemble_library.get_unique_molecules():
    #     rdkit_mol = ce.mol
    #     bioactive_conf = rdkit_mol.GetConformer(0)
    #     pdb_id = bioactive_conf.GetProp('PDB_ID')
    #     generated_conf_ids = [conf.GetId()
    #                           for conf in rdkit_mol.GetConformers()
    #                           if conf.HasProp('Generator')]
            
    #     try :
    #         ccdc_mols = [ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_mol,
    #                                                                 conf_id=conf_id)
    #                     for conf_id in generated_conf_ids]
    #         if len(ccdc_mols) == 100 :
    #             pdbbind_docking.dock_molecule_conformations(ccdc_mols, pdb_id)
    #     except KeyboardInterrupt :
    #         sys.exit(0)
    #     except :
    #         print(f'Docking failed for {pdb_id}')
            
    # params = []
    # for i, smiles in enumerate(test_smiles) :
    #     try :
    #         ce = conf_ensemble_library.get_conf_ensemble(smiles)
    #         params.append((i, smiles, ce))
    #     except :
    #         print('SMILES not in conf_ensemble_library')
    # with Pool(12) as pool :
    #     pool.map(dock_smiles, params)
            
