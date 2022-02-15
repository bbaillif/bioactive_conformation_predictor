#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pickle
import time
import multiprocessing
from multiprocessing import Pool
import pandas as pd
import torch
import tempfile
import copy
import random

from tqdm import tqdm
from molecule_featurizer import MoleculeFeaturizer
from rdkit import Chem
from ccdc.docking import Docker
from ccdc.io import MoleculeReader, EntryReader
from ccdc_rdkit_connector import CcdcRdkitConnector
from ccdc.descriptors import MolecularDescriptors
from ccdc.entry import Entry
from torch_geometric.data import Batch
from litschnet import LitSchNet
from mol_viewer import MolViewer
from collections import defaultdict


def gold_docking(ccdc_mols, 
                 native_ligand,
                 protein_file: str,
                 ligand_file: str,
                 dock_id: str,
                 rigid: bool=True,
                 n_pose_per_conf: int=5,
                 diverse_solutions=True) :
    
    connector = CcdcRdkitConnector()
    
    docker = Docker()
    settings = docker.settings
    
    settings.add_protein_file(protein_file)
    settings.reference_ligand_file = ligand_file
    
    # Define binding site
    protein = settings.proteins[0]
    settings.binding_site = settings.BindingSiteFromLigand(protein, native_ligand, 8.0)
        
    if rigid :
        settings.fix_ligand_rotatable_bonds = 'all'
        
    ligand_prep = Docker.LigandPreparation()
        
    output_dir = os.path.abspath(os.path.join('gold_docking', f'{dock_id}'))
    if not os.path.exists(output_dir) :
        os.mkdir(output_dir)
        
    # Add conformations to dock
    for i, ccdc_mol in enumerate(ccdc_mols) :
        ccdc_mol = ligand_prep.prepare(Entry.from_molecule(ccdc_mol))
        mol2_file_path = os.path.join(output_dir, f'{dock_id}_{i}.mol2')
        mol2_string = ccdc_mol.to_string(format='mol2')
        with open(mol2_file_path, 'w') as writer :
            writer.write(mol2_string)
        settings.add_ligand_file(mol2_file_path, n_pose_per_conf)

    settings.fitness_function = 'plp'
    settings.autoscale = 10.
    settings.early_termination = False
    
    if diverse_solutions:
        settings.diverse_solutions = True
    
    settings.output_directory = output_dir
    settings.output_file = os.path.join(output_dir, f'docked_ligands_{dock_id}.mol2')

    start_time = time.time()
    results = docker.dock(file_name=os.path.join(output_dir, f'api_gold.conf'))
    print(results.return_code)
    runtime = time.time() - start_time

#     batch_conf_file = settings.conf_file
#     settings = Docker.Settings.from_file(batch_conf_file)
    
    results = Docker.Results(settings)
    ligands = results.ligands
    
    ligand_scores = np.array([l.fitness() for l in ligands])
    ligand_mols = [l.molecule for l in ligands]
    for mol in ligand_mols :
        mol.remove_atoms([atom for atom in mol.atoms if atom.atomic_number < 2])
    
    if len(ligand_mols[0].atoms) == len(native_ligand.atoms) :
    
        results_d = {}
        # find top score pose
        top_score_index = ligand_scores.argmax()
        top_score_pose = ligand_mols[top_score_index]

        # find best pose 
        rmsds_to_real_pose = np.array([MolecularDescriptors.rmsd(native_ligand, ligand_mol, overlay=False) for ligand_mol in ligand_mols])
        best_pose_index = rmsds_to_real_pose.argmin()
        best_pose = ligand_mols[best_pose_index]

        top_score = ligand_scores.max()
        min_rmsd = rmsds_to_real_pose.min()
        
        results_d['top_score_pose'] = connector.ccdc_mol_to_rdkit_mol(top_score_pose)
        results_d['top_score'] = top_score
        results_d['best_pose'] = connector.ccdc_mol_to_rdkit_mol(best_pose)
        results_d['min_rmsd'] = min_rmsd
        results_d['runtime'] = runtime
        results_d['docking_power'] = MolecularDescriptors.rmsd(native_ligand, top_score_pose, overlay=False) < 2
        
        return results_d
    
    else :
        raise Exception('Docking was not successful')


def dock_smiles(params) :
    dock_id, smiles, ce = params
    connector = CcdcRdkitConnector()
    pdbbind_refined_dir = '/home/benoit/PDBBind/PDBbind_v2020_refined/refined-set/'
    pdbbind_general_dir = '/home/benoit/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/'
    #ce = cel.get_conf_ensemble(smiles)
    pdb_id = ce.mol.GetConformer(0).GetProp('PDB_ID')
    generated_ids = [i for i, conf in enumerate(ce.mol.GetConformers()) if conf.HasProp('Generator')]
    n_generated_confs = len(generated_ids)
    
    if n_generated_confs == 100 :
    
        if pdb_id in os.listdir(pdbbind_refined_dir) :
            protein_file = os.path.join(pdbbind_refined_dir, pdb_id, f'{pdb_id}_protein.pdb')
            ligand_file = os.path.join(pdbbind_refined_dir, pdb_id, f'{pdb_id}_ligand.mol2')
        else :
            protein_file = os.path.join(pdbbind_general_dir, pdb_id, f'{pdb_id}_protein.pdb')
            ligand_file = os.path.join(pdbbind_general_dir, pdb_id, f'{pdb_id}_ligand.mol2')

        native_ligand = connector.rdkit_conf_to_ccdc_mol(ce.mol, conf_id=0)
            
        ccdc_mols = []
        for conf_id in generated_ids :
            ccdc_mols.append(connector.rdkit_conf_to_ccdc_mol(ce.mol, conf_id=conf_id))

        try :
            
            all_results_d = {}
            
            all_results_d['pdb_id'] = pdb_id
            all_results_d['smiles'] = smiles
            
            # Flexible native ligand
            current_dock_id = f'{dock_id}_flexible'
            all_results_d['flexible'] = gold_docking(ccdc_mols=ccdc_mols[:1],
                                                     native_ligand=native_ligand,
                                                     protein_file=protein_file,
                                                     ligand_file=ligand_file,
                                                     dock_id=current_dock_id,
                                                     rigid=False,
                                                    n_pose_per_conf=20)
            
            # All rigid
            current_dock_id = f'{dock_id}_all'
            all_results_d['all'] = gold_docking(ccdc_mols=ccdc_mols,
                                                native_ligand=native_ligand,
                                                protein_file=protein_file,
                                                ligand_file=ligand_file, 
                                                dock_id=current_dock_id)
            
            # CCDC top 20
            current_dock_id = f'{dock_id}_ccdc'
            all_results_d['ccdc'] = gold_docking(ccdc_mols=ccdc_mols[:20],
                                                native_ligand=native_ligand,
                                                protein_file=protein_file,
                                                ligand_file=ligand_file,
                                                dock_id=current_dock_id)
            
            # Model top 20

            dummy_mol = copy.deepcopy(ce.mol)
            dummy_mol.RemoveAllConformers()
            for conf_id in generated_ids :
                dummy_mol.AddConformer(ce.mol.GetConformer(conf_id), assignId=True)
            
            data_list = mol_featurizer.featurize_mol(dummy_mol)
            batch = Batch.from_data_list(data_list)
            
            with torch.no_grad() :
                preds = litschnet(batch).cpu().numpy()
            preds = preds.reshape(-1)
            top20_index = preds.argsort()[:20]
            sorted_ccdc_mols = [ccdc_mol for i, ccdc_mol in enumerate(ccdc_mols) if i in top20_index]
            
            current_dock_id = f'{dock_id}_model'
            all_results_d['model'] = gold_docking(ccdc_mols=sorted_ccdc_mols,
                                                    native_ligand=native_ligand,
                                                    protein_file=protein_file,
                                                    ligand_file=ligand_file,
                                                    dock_id=current_dock_id)
            
            # Energy top 20
            energies = np.array([data.energy for data in data_list])
            top20_index = energies.argsort()[:20]
            sorted_ccdc_mols = [ccdc_mol for i, ccdc_mol in enumerate(ccdc_mols) if i in top20_index]
            
            current_dock_id = f'{dock_id}_energy'
            all_results_d['energy'] = gold_docking(ccdc_mols=sorted_ccdc_mols,
                                                  native_ligand=native_ligand,
                                                  protein_file=protein_file,
                                                  ligand_file=ligand_file,
                                                dock_id=current_dock_id)

            # Random top 20
            current_dock_id = f'{dock_id}_random'
            random.shuffle(ccdc_mols)
            all_results_d['random'] = gold_docking(ccdc_mols=ccdc_mols[:20],
                                                native_ligand=native_ligand,
                                                protein_file=protein_file,
                                                ligand_file=ligand_file,
                                                dock_id=current_dock_id)
            
            with open(os.path.join('gold_docking', f'{dock_id}_results.p'), 'wb') as f :
                pickle.dump(all_results_d, f)
            
        except Exception as e :
            print(f'Docking failed for {smiles}')
            print(str(e))


if __name__ == '__main__':
    
    data_dir = 'data/'

    with open(os.path.join(data_dir, 'scaffold_splits', f'test_smiles_scaffold_split_0.txt'), 'r') as f :
        test_smiles = f.readlines()
        test_smiles = [smiles.strip() for smiles in test_smiles]

    with open(os.path.join(data_dir, 'raw', 'ccdc_generated_conf_ensemble_library.p'), 'rb') as f :
        cel = pickle.load(f)

    mol = cel.get_conf_ensemble('COc1ccc(-c2cn(C)c(=O)c3cc(C(=O)NC4CCS(=O)(=O)CC4)sc23)cc1OC').mol

    encoder_path = os.path.join(data_dir, 'molecule_encoders.p')
    if os.path.exists(encoder_path) : # Load existing encoder
        with open(encoder_path, 'rb') as f:
            mol_encoders = pickle.load(f)
    mol_featurizer = MoleculeFeaturizer(mol_encoders)

    experiment_name = f'scaffold_split_0_new'
    if experiment_name in os.listdir('lightning_logs') :
        checkpoint_name = os.listdir(os.path.join('lightning_logs', experiment_name, 'checkpoints'))[0]
        checkpoint_path = os.path.join('lightning_logs', experiment_name, 'checkpoints', checkpoint_name)
        litschnet = LitSchNet.load_from_checkpoint(checkpoint_path=checkpoint_path)
    litschnet.eval()
    
#     for i, smiles in enumerate(test_smiles) :
#         dock_smiles((i, smiles, cel.get_conf_ensemble(smiles)))
    
    params = []
    for i, smiles in enumerate(test_smiles) :
        try :
            ce = cel.get_conf_ensemble(smiles)
            params.append((i, smiles, ce))
        except :
            print('SMILES not in CEL')
    with Pool(12) as pool :
        pool.map(dock_smiles, params)