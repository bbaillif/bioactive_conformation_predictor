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
os.makedirs(figures_dir_path, exist_ok=True)

pdbbind_processor = PDBBindMetadataProcessor()
pdbbind_mols = pdbbind_processor.get_molecules()
pdbbind_CEL = ConfEnsembleLibrary.from_mol_list(pdbbind_mols)
pdbbind_CEL_path = os.path.join(data_dir_path, 
                                'raw', 
                                'pdbbind_cel.p')
pdbbind_CEL.save()

os.makedirs(figures_dir_path, exist_ok=True)

pdbbind_smiles = [smiles for smiles, ce in pdbbind_CEL.get_unique_molecules()]
print(f'There are {len(pdbbind_smiles)} different smiles in PDBBind')

lines = []
for smiles, ce in pdbbind_CEL.get_unique_molecules() :
    # For computational (conformation generation) and dataset matching reasons,
    # PDBBind molecules having more than 50 heavy atoms were excluded
    included = ce.mol.GetNumHeavyAtoms() <= 50
    for conf in ce.mol.GetConformers() :
        lines.append([smiles, 'pdbbind', conf.GetProp('pdbbind_id'), included])

df_columns = ['smiles', 'dataset', 'id', 'included']
smiles_df = pd.DataFrame(lines, columns=df_columns)

excluded_smiles = smiles_df[~smiles_df['included']]['smiles'].unique()
print(f' There are {len(excluded_smiles)} excluded smiles')

# Heavy atom analysis
pdbbind_n_heavy_atoms = [ce.mol.GetNumHeavyAtoms() 
                         for smiles, ce in pdbbind_CEL.get_unique_molecules()]
sns.kdeplot(pdbbind_n_heavy_atoms, label='PDBBind')
plt.savefig(os.path.join(figures_dir_path, 'n_heavy_atoms_datasets'))

smiles_df.to_csv(os.path.join(data_dir_path, 'smiles_df.csv'))