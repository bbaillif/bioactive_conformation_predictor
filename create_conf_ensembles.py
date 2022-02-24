import pandas as pd
import os
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from conf_ensemble import ConfEnsembleLibrary
from rdkit import Chem
from rdkit import RDLogger                                                                                                                                                               

# Disable the warning when mol2 files are not parsable
RDLogger.DisableLog('rdApp.*') 

# To be able to save conformer properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

parser = argparse.ArgumentParser(description='Select data where conf ensembles will be created')
parser.add_argument('--data_dir', 
                    type=str, 
                    default='data/',
                    help='directory where data are stored')
parser.add_argument('--figures_dir', 
                    type=str,
                    default='figures/',
                    help='where figures are stored')

args = parser.parse_args()
data_dir_path = args.data_dir
figures_dir_path = args.figures_dir

# Extract PDBBind conformations
pdbbind_refined_dir_path = '../PDBBind/PDBbind_v2020_refined/refined-set/'
pdbbind_general_dir_path = '../PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/'

# defined using the metadata in the INDEX_general_PL_data file
widths = [6, 6, 7, 6, 17, 9, 200]
cols = ['PDB code', 
        'resolution', 
        'release year',
        '-logKd/Ki',
        'Kd/Ki',
        'reference',
        'ligand name']

pl_data_path = os.path.join(pdbbind_refined_dir_path, 
                            'index', 
                            'INDEX_general_PL_data.2020')
pl_data = pd.read_fwf(pl_data_path, widths=widths, skiprows=6, header=None)
pl_data.columns=cols
pl_data = pl_data[~pl_data['ligand name'].str.contains('-mer')]
pl_data.shape

correct_pdb_ids = pl_data['PDB code'].values

def extract_pdbbind_mols(directory_path, query_pdb_ids) :
    
    mols = []
    dirnames = os.listdir(directory_path)
    pdb_ids = [pdb_id for pdb_id in dirnames if pdb_id in query_pdb_ids]
    
    for pdb_id in pdb_ids :
        mol2_path = os.path.join(directory_path, 
                                 pdb_id, 
                                 f'{pdb_id}_ligand.mol2')
        try :
            mol = Chem.rdmolfiles.MolFromMol2File(mol2_path)
            if mol is not None :
                rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                if rdmol is not None : #rdkit parsable
                    #mol = PropertyMol(mol)
                    mol.GetConformer().SetProp('PDB_ID', pdb_id)
                    mols.append(mol)
                else :
                    print(f'{pdb_id} Not RDKit parsable')
        except :
            print('Impossible to read mol2 file for ' + pdb_id)
            
    return mols

general_mols = extract_pdbbind_mols(pdbbind_general_dir_path, correct_pdb_ids)
print(f'There are {len(general_mols)} mols in PDBBind general')
pdbbind_general_mols_path = os.path.join(data_dir_path, 
                                         'pdbbind_general_mol_list.p')
with open(pdbbind_general_mols_path , 'wb') as f :
    pickle.dump(general_mols, f)
    
general_CEL = ConfEnsembleLibrary(general_mols)
pdbbind_general_cel_path = os.path.join(data_dir_path, 
                                        'raw', 
                                        'pdbbind_general_cel.p')
with open(pdbbind_general_cel_path, 'wb') as f :
    pickle.dump(general_CEL, f)

refined_mols = extract_pdbbind_mols(pdbbind_refined_dir_path, correct_pdb_ids)
pdbbind_refined_mols_path = os.path.join(data_dir_path, 
                                         'pdbbind_refined_mol_list.p')
with open(pdbbind_refined_mols_path , 'wb') as f :
    pickle.dump(refined_mols, f)
    
refined_CEL = ConfEnsembleLibrary(refined_mols)
pdbbind_refined_cel_path = os.path.join(data_dir_path, 
                                        'raw', 
                                        'pdbbind_refined_cel.p')
with open(pdbbind_refined_cel_path, 'wb') as f :
    pickle.dump(refined_CEL, f)

pdbbind_CEL = general_CEL.merge(refined_CEL)
os.makedirs(os.path.join(data_dir_path, 'raw'), exist_ok=True)
pdbbind_cel_path = os.path.join(data_dir_path, 
                                'raw', 
                                'pdbbind_cel.p')
with open(pdbbind_cel_path, 'wb') as f :
    pickle.dump(pdbbind_CEL, f)
    
# Extract Platinum conformations
platinum_dataset_path = os.path.join(data_dir_path, 
                                     'platinum-dataset-2017-01-sdf', 
                                     'platinum_dataset_2017_01.sdf')
sdsupplier = Chem.rdmolfiles.SDMolSupplier(platinum_dataset_path)
platinum_mols = [mol for mol in sdsupplier]
print(f'There are {len(platinum_mols)} molecules in platinum')

def extract_platinum_pdb_ids(platinum_dataset_path) :

    with open(platinum_dataset_path, 'r') as f :
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    pdb_ids = []
    new_mol = True
    for line in lines :
        if new_mol :
            pdb_ids.append(line)
            new_mol = False
        if line == '$$$$' :
            new_mol = True
                
    return pdb_ids
    
# Add PDB_ID information to conformer
pdb_ids = extract_platinum_pdb_ids(platinum_dataset_path)
        
for i, mol in enumerate(platinum_mols) :
    mol.GetConformer().SetProp('PDB_ID', pdb_ids[i])
    
platinum_CEL = ConfEnsembleLibrary(platinum_mols)
platinum_CEL_path = os.path.join(data_dir_path, 
                                 'raw', 
                                 'platinum_cel.p')
with open(platinum_CEL_path, 'wb') as f :
    pickle.dump(platinum_CEL, f)
    
# Join PDBBind and Platinum, and compare the 2 datasets
all_CEL = pdbbind_CEL.merge(platinum_CEL)
all_CEL_path = os.path.join(data_dir_path, 'raw', 'all_conf_ensemble_library.p')
with open(all_CEL_path, 'wb') as f :
    pickle.dump(all_CEL, f)
print(f"""Total of {all_CEL.get_num_molecules()} different molecules 
      in PDBBind and Platinum""")

os.makedirs(figures_dir_path, exist_ok=True)

all_smiles = [smiles for smiles, ce in all_CEL.get_unique_molecules()]
pdbbind_smiles = [smiles for smiles, ce in pdbbind_CEL.get_unique_molecules()]
platinum_smiles = [smiles for smiles, ce in platinum_CEL.get_unique_molecules()]
print(f'There are {len(pdbbind_smiles)} different smiles in PDBBind')
print(f'There are {len(platinum_smiles)} different smiles in Platinum')

smiles_df = pd.DataFrame(all_smiles, columns=['smiles'])
smiles_df['pdbbind'] = smiles_df['smiles'].isin(pdbbind_smiles)
smiles_df['platinum'] = smiles_df['smiles'].isin(platinum_smiles)

pdbbind_n_heavy_atoms = [ce.mol.GetNumHeavyAtoms() 
                         for smiles, ce in pdbbind_CEL.get_unique_molecules()]
platinum_n_heavy_atoms = [ce.mol.GetNumHeavyAtoms() 
                          for smiles, ce in platinum_CEL.get_unique_molecules()]
sns.kdeplot(pdbbind_n_heavy_atoms, label='PDBBind')
sns.kdeplot(platinum_n_heavy_atoms, label='Platinum')
plt.savefig(os.path.join(figures_dir_path, 'n_heavy_atoms_datasets'))

# For computational (conformation generation) and dataset matching reasons,
# PDBBind molecules having more than 50 heavy atoms were excluded
included_smiles = [smiles 
                   for smiles, ce in all_CEL.get_unique_molecules() 
                   if ce.mol.GetNumHeavyAtoms() <= 50]
smiles_df['included'] = smiles_df['smiles'].isin(included_smiles)
len(included_smiles)
smiles_df.to_csv(os.path.join(data_dir_path, 'smiles_df.csv'))