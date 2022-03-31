import pandas as pd
import os
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit import RDLogger
from conf_ensemble_library import ConfEnsembleLibrary
from pdbbind_metadata_processor import PDBBindMetadataProcessor
from platinum_processor import PlatinumProcessor
                                                                                                                                                           

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
os.makedirs(os.path.join(data_dir_path, 'raw'), exist_ok=True)

pdbbind_processor = PDBBindMetadataProcessor()
pdbbind_mols = pdbbind_processor.get_molecules()
pdbbind_CEL = ConfEnsembleLibrary.from_mol_list(pdbbind_mols)
pdbbind_CEL_path = os.path.join(data_dir_path, 
                                'raw', 
                                'pdbbind_cel.p')
# pdbbind_CEL.save(pdbbind_CEL_path)
    
platinum_processor = PlatinumProcessor()
platinum_mols = []
for pdb_id, platinum_d in platinum_processor.available_structures.items() :
    for platinum_id, molecule in platinum_d.items() :
        platinum_mols.append(molecule)
    
platinum_CEL = ConfEnsembleLibrary.from_mol_list(platinum_mols)
platinum_CEL_path = os.path.join(data_dir_path, 
                                 'raw', 
                                 'platinum_cel.p')
# platinum_CEL.save(platinum_CEL_path)
    
# Join PDBBind and Platinum, and compare the 2 datasets
all_CEL = pdbbind_CEL.merge(platinum_CEL)
all_CEL_path = os.path.join(data_dir_path, 'raw', 'all_conf_ensemble_library.p')
with open(all_CEL_path, 'wb') as f :
    pickle.dump(all_CEL, f)
# ConfEnsembleLibrary.to_path(all_CEL, all_CEL_path)
all_CEL.save()
print(f"""Total of {all_CEL.get_num_molecules()} different molecules 
      in PDBBind and Platinum""")

os.makedirs(figures_dir_path, exist_ok=True)

all_smiles = [smiles for smiles, ce in all_CEL.get_unique_molecules()]
pdbbind_smiles = [smiles for smiles, ce in pdbbind_CEL.get_unique_molecules()]
platinum_smiles = [smiles for smiles, ce in platinum_CEL.get_unique_molecules()]
print(f'There are {len(pdbbind_smiles)} different smiles in PDBBind')
print(f'There are {len(platinum_smiles)} different smiles in Platinum')

lines = []
for smiles, ce in all_CEL.get_unique_molecules() :
    # For computational (conformation generation) and dataset matching reasons,
    # PDBBind molecules having more than 50 heavy atoms were excluded
    included = ce.mol.GetNumHeavyAtoms() <= 50
    for conf in ce.mol.GetConformers() :
        if conf.HasProp('pdbbind_id') :
            lines.append([smiles, 'pdbbind', conf.GetProp('pdbbind_id'), included])
        elif conf.HasProp('platinum_id') :
            lines.append([smiles, 'platinum', conf.GetProp('platinum_id'), included])

df_columns = ['smiles', 'dataset', 'id', 'included']
smiles_df = pd.DataFrame(lines, columns=df_columns)

faulty_smiles = ['O=C[Ru+9]12345(C6=C1C2C3=C64)n1c2ccc(O)cc2c2c3c(c4ccc[n+]5c4c21)C(=O)NC3=O',
 'Cc1cc2c3c(c4c5ccccc5n5c4c2[n+](c1)[Ru+9]51246(Cl)C5=C(C(=O)[O-])C1=C2C4=C56)C(=O)NC3=O']
# cannot be pickled because of a number of radical electron error
faulty_smiles_idx = smiles_df[smiles_df['smiles'].isin(faulty_smiles)].index
smiles_df.loc[faulty_smiles_idx, 'included'] = False

excluded_smiles = smiles_df[~smiles_df['included']]['smiles'].unique()
print(f' There are {len(excluded_smiles)} excluded smiles')

# Heavy atom analysis
pdbbind_n_heavy_atoms = [ce.mol.GetNumHeavyAtoms() 
                         for smiles, ce in pdbbind_CEL.get_unique_molecules()]
platinum_n_heavy_atoms = [ce.mol.GetNumHeavyAtoms() 
                          for smiles, ce in platinum_CEL.get_unique_molecules()]
sns.kdeplot(pdbbind_n_heavy_atoms, label='PDBBind')
sns.kdeplot(platinum_n_heavy_atoms, label='Platinum')
plt.savefig(os.path.join(figures_dir_path, 'n_heavy_atoms_datasets'))

def extract_pdb_id(s) :
    if len(s) == 4 :
        return s
    else :
        return s.split('_')[1].lower()
smiles_df['pdb_id'] = smiles_df['id'].apply(extract_pdb_id)
smiles_df.to_csv(os.path.join(data_dir_path, 'smiles_df.csv'))
