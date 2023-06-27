# Applying atomistic neural networks to bias conformer ensembles towards bioactive-like conformations

## Installation
Install the conda environment from the bioconfpred.yml file:
`conda env create -f bioconfpred.yml`

The conformer generation step, RMSD calculation and GOLD docking requires the tools from CCDC through the [CSD Python API](https://downloads.ccdc.cam.ac.uk/documentation/API/) that can be [installed using conda](https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html#id2) (license required).
You can replace these parts in the code by open source solutions like RDKit ([EmbedMultipleConfs](https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.EmbedMultipleConfs) and [GetBestRMSD](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html#rdkit.Chem.rdMolAlign.GetBestRMS)) and [Vina](https://github.com/ccsb-scripps/AutoDock-Vina) (however I am not aware how to keep torsions rigid for the re-docking).

## Reproducing paper results
To reproduce the results in the manuscript, the first step is to change the parameter values in params.py to match your desktop pathes. The ROOT_DIRPATH is your data directory (e.g. HDD), the PDBBIND_GENERAL_URL and PDBBIND_REFINED_URL are the download URL of PDBbind (login to their website and copy-paste the linksfrom the Download page). Double check if all the remaining pathes are new for youto avoid any overwrite.

Activate the conda environment:
`conda activate bioconfpred`

The next step is to train models:
`python train_models.py`

Each data source (i.e. PDBbind, ChEMBL, ENZYME) will be downloaded and pre-processed,and splits will be made. The scripts are built to train 5 models for each splitting strategy (random and scaffold).

The next step is to run the model evaluations:
`python evalute_models.py`

The computed results can be visualized using the jupyter notebook compare_performances.ipynb

The docking part is run with
`python pdbbind_docking.py`

Docking results can be analyzed with analyze_targets_pdbbind_docking.ipynb

You can also download the data generated on my desktop [here](https://figshare.com/articles/dataset/Data_for_Applying_atomistic_neural_networks_to_bias_conformer_ensemble_towards_bioactive-like_conformations/23580267).
In the data folder, you can find:
- pdb_conf_ensembles: Ensembles of bioactive conformation for each unique ligand.
- gen_conf_ensembles: Ensembles of generated conformers for each unique ligand.
- rmsds: ARMSD between each bioactive conformation and generated conformer.
- processed: Pytorch Geometric processed data
- splits: Random and Scaffold splits (each unique ligand represented by SMILES)
- lightning_logs: Logs of model training (using Pytorch Lightning)
- pyg_mol_ids.csv: Allow to retrieve the original ligand-conformation for each data in the Pytorch Geometric dataset

One pretrained model for each atomistic neural network and data split are available [here](https://figshare.com/articles/dataset/Pretrained_atomistic_neural_networks/23586240)

## Architecture
The package is built as follow.

conf_ensemble directory contains classes to handle conformer ensembles
- ConfEnsemble is a wrapper around a RDKit Mol that handles the differentconformers as Conformer in the Mol, making sure that all atoms in each conformer are matched, and handling properties in each conformer
- ConfEnsembleLibrary is a wrapper around a dict that links an ensemble name to a ConfEnsemble. Each library has a directory where each ensemble can be stored, and also storing metadata in ensemble_names.csv and pdb_df.csv
- RMSDCalculator (to compute RMSD between bioactive and generated conformers)

data contains classes to manage data
- dataset contains the ConfEnsembleDataset base class, and the PyGDataset subclass (and the MoleculeEncoders to encode atom and bond data)
- featurizer contains the base class MolFeaturizer, and the PyGFeaturizer subclass
- preprocessing contains the ConfGenerator (wrapping the CSD conformer generator) and MolStandardizer (standardize input ligand)
- split contains the DataSplit base class, the MoleculeSplit subclass and RandomSplit and ScaffoldSplit subclasses of MoleculeSplit
- utils contains database connectors and other useful classes: ChEMBL, ENZYME, LigandExpo, PDBbind, MolConverter, SimilaritySearch
- pose_reader.py to read output poses from GOLD

docking contains classes to run docking with GOLD: GOLDDocker and PDBbindDocking

evaluator contains classes to evaluate rankers/models: Evaluator base class, and ConfEnsembleModelEvaluator and RankerEvaluator subclasses

model contains classes to build atomistic neural networks:
- atomistic contains core models: ConfPredModel base class, AtomisticNN class, and AtomicSchNet, AtomicDimeNet and AtomicComENet subclasses. These are the core models.
- other classes are AtomisticNNModel base class, SchNetModel, DimeNetModel and ComENetModel subclasses, that embeds all functions needed to predict the ARMSD from input conformation

rankers contains classes to rank conformers based on predicted ARMSD or baselines: ConfRanker base class, and subclasses RandomRanker (randomly shuffling conformers), NoRankingRanker (keep original order, e.g. CSD conformer generator order), EnergyRanker (ascending MMFF94s energy), SASARanker (descending Solvent Accessible Surface Area), RGyrRanker (descending Radius of Gyration), TFD2SimRefMCSRanker (ascending TFD of the MCS to the most similar training molecule) and ModelRanker (ascending model ARMSD prediction)

utils contains the MolConfViewer, based on nglviewer, useful to visualize molecules in a Jupyter notebook

## Basic usage

The main purpose of the package is to rank conformer ensemble to obtain a higher rate of bioactive-like conformers in early ranks. 

You can use a trained model to fuel a ranker that can give you the ranks of each conformer in a molecule:
```python
from models import ComENetModel
from rankers import ModelRanker
checkpoint_path = /path/to/your/favorite/model_checkpoint.p
model = ComENetModel.load_from_checkpoint(checkpoint_path) 
```

Alternatively, you can use the data_split to load the checkpoints of a trained model
```python
from data.split import RandomSplit
data_split = RandomSplit() # default is the split number 0
model = ComENetModel.get_model_for_data_split(data_split)
```

Note: You can also use  

```python
ranker = ModelRanker(model)
mol = yourFavoriteMoleculeWithConformers
ranks = ranker.rank_molecule(mol)
```

Other atomistic neural networks are available (SchNetModel and DimeNetModel) but they lead to lower ranking performances, and other rankers can be used as baselines (e.g. EnergyRanker, TFD2SimRefMCSRanker)

In case you have different conformers of the same molecule in different RDKit molecules, you can create a conformer ensemble from a list of RDKit Mol
```python
from conf_ensemble import ConfEnsemble
mol_list = listOfConformersForTheSameMolecularGraph # including same chirality
ce = ConfEnsemble(mol_list) # ce = conf ensemble
mol = ce.mol # mol is stored in the ce
```

You can create a conformer ensemble library from a list of molecule or a dictionnary {name: mol_list} (default name is SMILES):
```python
from conf_ensemble_library import ConfEnsembleLibrary
mol_list = listOfConformersForAnyMolecularGraph
cel = ConfEnsembleLibrary.from_mol_list(mol_list) # cel = conf ensemble library
# if names is not given as argument of cel, the SMILES will be used
```

For more methods of the ranker objects, please see the ConfRanker base class, the RankerEvaluator and the evaluate_rankers.py

## Details

MolStandardizer uses MolVS which removes hydrogens. Don't use on ligand structure if you want to keep hydrogens.
