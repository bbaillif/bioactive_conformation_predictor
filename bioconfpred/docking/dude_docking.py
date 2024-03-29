import os
import sys
import time
import numpy as np
import argparse
import json

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.ML.Scoring.Scoring import CalcEnrichment, CalcBEDROC
from multiprocessing import Pool
from .gold_docker import GOLDDocker
from bioconfpred.conf_ensemble import ConfEnsemble
from bioconfpred.model import SchNetModel
from bioconfpred.ranker import (ModelRanker, 
                                RandomRanker,
                                NoRankingRanker,
                                EnergyRanker,
                                PropertyRanker,
                                ConfRanker)
from bioconfpred.data.split import RandomSplit
from bioconfpred.data.utils import MolConverter
from typing import List, Tuple, Dict

try:
    from ccdc.io import MoleculeReader, Molecule
    from ccdc.conformer import ConformerGenerator
except:
    print('CSD Python API not installed')

class DUDEDocking() :
    """ Performs docking for actives and decoys for a target in DUD-E
    Generated 20 poses per molecule, and compares virtual screening performances
    for different selectors (model, energy, random...)
    
    :param target: Target to dock
    :type target: str
    :param dude_dir: Directory path to DUD-E data
    :type dude_dir: str
    :param rigid_docking: True to perform rigid-ligand docking, False for flexible
    :type rigid_docking: bool
    :param use_cuda: True to use cuda for DL model scoring, False to use CPU
    :type use_cuda: bool
    """
    
    def __init__(self,
                 target: str = 'jak2',
                 dude_dir: str = '/home/bb596/hdd/DUD-E/all',
                 rigid_docking: bool = False,
                 use_cuda: bool = True):
        
        self.target = target
        self.dude_dir = dude_dir
        self.rigid_docking = rigid_docking
        self.use_cuda = use_cuda
        
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
            
        self.mol_converter = MolConverter()
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
                                      output_dir='/home/bb596/hdd/gold_docking_dude',
                                      experiment_id=self.experiment_id,
                                      prepare_protein=True)
        
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
                
                
    def dock_pool(self, 
                  max_f_actives: int = 1) :
        """
        Dock molecules using multi-threading
        
        :param max_f_actives: Fraction of actives to dock. Used to dock only a 
            part of the actives
        :type max_f_actives: int
        """
        
        active_mols_reader = MoleculeReader(self.actives_path)
        active_mols = [mol for mol in active_mols_reader]
        n_actives = int(len(active_mols) * max_f_actives)
        active_mols = active_mols[:n_actives]
        active_mol_ids = [f'active_{i}' for i in range(n_actives)]
        
        decoy_mols_reader = MoleculeReader(self.decoys_path)
        # n_decoys = 50 * n_actives
        decoy_mols = [mol for mol in decoy_mols_reader] # [:n_decoys]
        decoy_mol_ids = [f'decoy_{i}' for i in range(len(decoy_mols))]
        
        all_mols = active_mols + decoy_mols
        mol_ids = active_mol_ids + decoy_mol_ids
        
        rdkit_mols = [self.mol_converter.ccdc_mol_to_rdkit_mol(ccdc_mol=ccdc_mol)
                      for ccdc_mol in all_mols]
        
        print(f'Number of threads : {len(mol_ids)}')
        params = [(mol_id, rdkit_mol)
                  for mol_id, rdkit_mol
                  in zip(mol_ids, rdkit_mols)]
        with Pool(processes=12, maxtasksperchild=1) as pool :
            pool.map(self.dock_thread, params)
                
                
    def dock_thread(self,
                    params: Tuple[str, Mol]) :
        """
        Dock a single molecule, thread used in dock_pool
        
        :param params: Parameters for the thread: the mol_id and the RDKit mol
            to dock
        :type params: (str, Mol)
        """
        mol_id, rdkit_mol = params
        try :
            ccdc_mol = self.mol_converter.rdkit_conf_to_ccdc_mol(rdkit_mol=rdkit_mol)
            
            if self.rigid_docking :
                ccdc_mols = self.generate_conformers_for_molecule(ccdc_mol,
                                                                  n_confs=250)
            else :
                ccdc_mols = [ccdc_mol]
                
            n_poses = 5
            
            self.gold_docker = GOLDDocker(protein_path=self.protein_path,
                                        native_ligand_path=self.ligand_path,
                                        output_dir='/home/bb596/hdd/gold_docking_dude',
                                        experiment_id=self.experiment_id,
                                        prepare_protein=True) 
            results = self.gold_docker.dock_molecules(ccdc_mols=ccdc_mols,
                                                    mol_id=mol_id,
                                                    n_poses=n_poses,
                                                    rigid=self.rigid_docking)
        except :
            print(f'Docking failed for {mol_id}')
    
    
    def generate_conformers_for_molecule(self,
                                         ccdc_mol: Molecule,
                                         n_confs: str = 250) -> List[Molecule]:
        """
        Generate conformers using the CCDC conformer generator for given input
        CCDC Molecule
        
        :param ccdc_mol: Molecule to generate conformers for
        :type ccdc_mol: Molecule
        :param n_confs: Maximum number of conformers to generate
        :type n_confs: int
        
        """
        conf_generator = ConformerGenerator()
        conf_generator.settings.max_conformers = n_confs
        conformers = conf_generator.generate(ccdc_mol)
        return [conf.molecule for conf in conformers]
           
                
    def ef_analysis(self) :
        """Analyse virtual screening results depending on the ranker used
        to select each molecule pose. Evaluated rankers are model (DL model
        for bioactive RMSD prediction), energy, score and random.
        
        """
        
        self.gold_docker = GOLDDocker(protein_path=self.protein_path,
                                      native_ligand_path=self.ligand_path,
                                      output_dir='/home/bb596/hdd/gold_docking_dude',
                                      experiment_id=self.experiment_id)
        
        data_split = RandomSplit()
        checkpoint_path = data_split.get_bioschnet_checkpoint_path()
        config = {"num_interactions": 6,
                  "cutoff": 10,
                  "lr":1e-5,
                  'batch_size': 256,
                  'data_split': data_split}
        model = SchNetModel.load_from_checkpoint(checkpoint_path, 
                                               config=config)
        
        self.rankers: List[ConfRanker] = [
            ModelRanker(model=model, use_cuda=self.use_cuda),
            RandomRanker(),
            NoRankingRanker(),
            EnergyRanker(),
            # PropertyRanker(descriptor_name='Gold.PLP.Fitness',
            #                ascending=False)
        ]
        
        self.conf_fractions = np.around(np.arange(0.01, 1.01, 0.01), 2)
        self.ef_fractions = np.around(np.arange(0.01, 1.01, 0.01), 2)
        
        self.active_max_scores = [] # [{percentage_conf : max_score}]
        self.decoy_max_scores = []
        
        dude_docking_dir = os.path.join(self.gold_docker.output_dir,
                                        self.gold_docker.experiment_id)
        docked_dirs = os.listdir(dude_docking_dir)
        
        # Only test flexible what is in rigid
        if self.rigid_docking == False:
            dude_docking_rigid_dir = os.path.join(self.gold_docker.output_dir,
                                                    self.gold_docker.experiment_id + '_rigid')
            rigid_dirs = os.listdir(dude_docking_rigid_dir)
            docked_dirs = [d for d in docked_dirs if d in rigid_dirs]
            
        active_dirs = [os.path.join(dude_docking_dir, d)
                       for d in docked_dirs 
                       if 'active' in d]
        decoy_dirs = [os.path.join(dude_docking_dir, d) 
                      for d in docked_dirs 
                      if 'decoy' in d]
        
        for active_dir in tqdm(active_dirs) :
            max_scores = self.evaluate_molecule(dirname=active_dir)
            if max_scores is not None :
                self.active_max_scores.append(max_scores)
                
        for decoy_dir in tqdm(decoy_dirs) :
            max_scores = self.evaluate_molecule(dirname=decoy_dir)
            if max_scores is not None :
                self.decoy_max_scores.append(max_scores)
               
        results = {} # selector_name, percentage, efs, fractions
        try:
            for ranker in self.rankers :
                selector_name_results = {}
                
                active_max_scores = [max_scores[ranker.name] 
                                    for max_scores in self.active_max_scores]
                decoy_max_scores = [max_scores[ranker.name] 
                                    for max_scores in self.decoy_max_scores]
                for conf_fraction in self.conf_fractions :
                    percentage_results = {}
                    active_2d_list = [[max_scores[conf_fraction], True] 
                                    for max_scores in active_max_scores]
                    decoy_2d_list = [[max_scores[conf_fraction], False] 
                                    for max_scores in decoy_max_scores]
                
                    all_2d_list = active_2d_list + decoy_2d_list
                    all_2d_array = np.array(all_2d_list)
                    
                    # if ranker.name == 'Gold.PLP.Fitness' or not self.use_selector_scoring:
                    #     sorting = np.argsort(-all_2d_array[:, 0]) # the minus is important here
                    # elif self.use_selector_scoring :
                    #     sorting = np.argsort(all_2d_array[:, 0])
                    sorting = np.argsort(-all_2d_array[:, 0])
                    sorted_2d_array = all_2d_array[sorting]

                    # efs = CalcEnrichment(sorted_2d_array, 
                    #                     col=1,
                    #                     fractions=self.ef_fractions)
                    # efs = np.around(efs, decimals=3)
                    # ef_results = {}
                    # for ef_fraction, ef in zip(self.ef_fractions, efs) :
                    #     # print(f'{selector_name} has EF{fraction} of {ef}')
                    #     ef_results[ef_fraction] = ef
                    # percentage_results['ef'] = ef_results
                    
                    bedroc = CalcBEDROC(sorted_2d_array, 
                                        col=1, 
                                        alpha=20)
                    bedroc = np.around(bedroc, decimals=3)
                    # print(f'{selector_name} has BEDROC of {bedroc}')
                    
                    percentage_results['all_2d_list'] = all_2d_list
                    percentage_results['bedroc'] = bedroc
                    
                    selector_name_results[conf_fraction] = percentage_results
                    
                results[ranker.name] = selector_name_results
                
        except Exception as e:
            print(str(e))
            import pdb;pdb.set_trace()
            
        results_file_name = 'results_docking_score_percentage.json'
        results_path = os.path.join(self.gold_docker.settings.output_directory,
                                    results_file_name)
        print(results_path)
        with open(results_path, 'w') as f :
            json.dump(results, f)
        
        
    def evaluate_molecule(self,
                          dirname: str) -> Dict[str, Dict[float, float]]:
        """
        Evaluate the ranking of poses for a molecule using setup rankers
        
        :param dirname: Directory where the docking results of the molecule
            are stored
        :type dirname: str
        :return: Results in a dictionnary: {Ranker name: {Fraction: Max PLP score}}
        :rtype: Dict[str, Dict[float, float]]
        
        """
        max_scores = {}
        # import pdb;pdb.set_trace()
        poses_ce = self.get_poses_ce(dirname)
        if poses_ce :
            for ranker in self.rankers:
                # print(ranker.name)
                ranked_poses = ranker.rank_confs(poses_ce.mol)
                if ranked_poses :
                    max_scores[ranker.name] = {}
                    n_poses = ranked_poses.GetNumConformers()
                    scores = []
                    for pose in ranked_poses.GetConformers() :
                        score = pose.GetProp('Gold.PLP.Fitness')
                        if isinstance(score, str):
                            score = score.strip()
                            score = float(score)
                        scores.append(score)
                    for fraction in self.conf_fractions :
                        try :
                            n_conf = int(np.ceil(n_poses * fraction))
                            subset_scores = scores[:n_conf]
                            max_score = np.max(subset_scores)
                            max_scores[ranker.name][fraction] = max_score
                        except :
                            import pdb;pdb.set_trace()
                else :
                    max_scores = None
                    break
        else :
            max_scores = None
    
        return max_scores
    

    def get_poses_ce(self, 
                  directory) -> ConfEnsemble:
        """Obtain poses for a given directory (referring to a molecule)
        
        :param directory: directory for GOLD docking of a single molecule
        :type directory: str
        :return: conf ensemble containing poses
        :rtype: ConfEnsemble
        """
        
        ce = None
        docked_ligands_path = os.path.join(directory,
                                           self.gold_docker.docked_ligand_name)
        if os.path.exists(docked_ligands_path) :
            try:
                ce = ConfEnsemble.from_file(docked_ligands_path)
                seen_ligands = []
                for pose in ce.mol.GetConformers() :
                    identifier = pose.GetProp('_Name')
                    lig_num = identifier.split('|')[1]
                    if not lig_num in seen_ligands :
                        seen_ligands.append(lig_num)
                    else:
                        conf_id = pose.GetId()
                        ce.mol.RemoveConformer(conf_id)
            except Exception as e :
                print(f'Reading poses failed for {docked_ligands_path}')
                print(str(e))
        
        return ce
        
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Perform docking of a DUD-E target')
    parser.add_argument('--target', 
                        type=str, 
                        default='jak2',
                        help='Target to dock')
    parser.add_argument('--rigid', 
                        action='store_true',
                        help='Generate conformations for rigid docking, else flexible')
    args = parser.parse_args()
    
    if args.target == 'all' :
        targets = os.listdir('/home/bb596/hdd/DUD-E/all')
    else :
        targets = [args.target]
        
    targets = ['cdk2', 'bace1'] #, 'nos1']
    for target in targets :
        # dude_docking = DUDEDocking(target=target,
        #                            rigid_docking=True)
        # dude_docking.dock_pool()
        # dude_docking.ef_analysis()
        dude_docking = DUDEDocking(target=target,
                                   rigid_docking=False)
        dude_docking.dock_pool()
        dude_docking.ef_analysis()