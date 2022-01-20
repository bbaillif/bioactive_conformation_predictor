from tkinter import N
from rdkit import Chem
from ccdc_rdkit_connector import CcdcRdkitConnector
from ccdc.conformer import ConformerGenerator
from tqdm import tqdm

import pickle
import os
import pandas as pd
import argparse

# To be able to save conformer properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', 
                    type=str, 
                    default='data/',
                    help='directory where data are stored')
parser.add_argument('--dataset',
                    type=str,
                    default='pdbbind',
                    help='dataset to generate conformations for')
args = parser.parse_args()
data_dir_path = args.data_dir

all_CEL_path = os.path.join(data_dir_path, 'raw', 'all_conf_ensemble_library.p')
with open(all_CEL_path, 'rb') as f :
    all_CEL = pickle.load(f)
    
ccdc_rdkit_connector = CcdcRdkitConnector()

smiled_df_path = os.path.join(data_dir_path, 'smiles_df.csv')
smiles_df = pd.read_csv(smiled_df_path, index_col=0)
included_smiles = smiles_df[smiles_df['included']
                            & smiles_df['pdbbind']
                            & ~smiles_df['platinum']].values

initial_ccdc_mols = []
corresponding_ce_mols = []
for smiles, conf_ensemble in tqdm(all_CEL.get_unique_molecules()) :
    if smiles in included_smiles : # see comments above
        ccdc_mol = ccdc_rdkit_connector.rdkit_conf_to_ccdc_mol(conf_ensemble.mol)
        assert conf_ensemble.mol.GetNumAtoms() == len(ccdc_mol.atoms)
        initial_ccdc_mols.append(ccdc_mol)
        corresponding_ce_mols.append(conf_ensemble.mol)
        
n_conf_per_chunk = 100
chunk_range = range(0, len(initial_ccdc_mols), n_conf_per_chunk)
chunk_idxs = [list(range(i, i + n_conf_per_chunk)) for i in chunk_range]
print(f'There are {len(initial_ccdc_mols)} initial CCDC mols')

ccdc_conformer_generator = ConformerGenerator(nthreads=12)
ccdc_conformer_generator.settings.max_conformers = 100

for chunk_idx in tqdm(chunk_idxs) :
    
    mol_list = [mol 
                for i, mol in enumerate(initial_ccdc_mols) 
                if i in chunk_idx]
    ce_mols = [mol 
               for i, mol in enumerate(corresponding_ce_mols) 
               if i in chunk_idx]
    
    conformers = ccdc_conformer_generator.generate(mol_list)
    
    for ce_mol, conformers in zip(ce_mols, conformers) :
        try :
            generated_conf_ids = ccdc_rdkit_connector.ccdc_conformers_to_rdkit_mol(conformers, ce_mol)
        except Exception as e :
            print(e)

ccdc_confs_CEL_unfiltered_path = os.path.join(data_dir_path, 
                                              'raw', 
                                              'ccdc_generated_cel_unfiltered.p')
with open(ccdc_confs_CEL_unfiltered_path, 'wb') as f :
    pickle.dump(all_CEL, f)
    
# here we only have the molecules parsed identically by RDKit (from mol2) and CSD (from smiles)
faulty_smiles = ['O=C[Ru+9]12345(C6=C1C2C3=C64)n1c2ccc(O)cc2c2c3c(c4ccc[n+]5c4c21)C(=O)NC3=O',
 'Cc1cc2c3c(c4c5ccccc5n5c4c2[n+](c1)[Ru+9]51246(Cl)C5=C(C(=O)[O-])C1=C2C4=C56)C(=O)NC3=O']
# cannot be pickled because of a number of radical electron error
smiles_df.loc[smiles_df['smiles'].isin(faulty_smiles), 'included'] = False
smiles_df.to_csv(os.path.join(data_dir_path, 'smiles_df.csv'))
excluded_smiles = smiles_df[~smiles_df['included']]['smiles'].values
print(f' There are {len(excluded_smiles)} excluded smiles')
for smiles in excluded_smiles :
    all_CEL.library.pop(smiles)
all_CEL.get_num_molecules()

ccdc_generated_CEL_path = os.path.join(data_dir_path, 
                                       'raw', 
                                       'ccdc_generated_cel.p')
with open(ccdc_generated_CEL_path, 'wb') as f :
    pickle.dump(all_CEL, f)

conf_list = []
for smiles in smiles_df[smiles_df['included']]['smiles'].values :
    confs = all_CEL.get_conf_ensemble(smiles).mol.GetConformers()
    for conf in confs :
        generated = 'Generator' in conf.GetPropsAsDict()
        conf_list.append([smiles, generated])
len(conf_list)
conf_df = pd.DataFrame(conf_list, columns=['smiles', 'generated'])
conf_df.to_csv(os.path.join(data_dir_path, 'conf_df.csv'))