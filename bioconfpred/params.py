import os

ROOT_DIRPATH = '/home/bb596/hdd/'
if not os.path.exists(ROOT_DIRPATH):
    print('You may want to define your own root directory, currently setting to "./"')
    ROOT_DIRPATH = './'

# These URL needs to be provided by the user, as PDBbind is under license
# and requires to be logged in. The "Cloud CDN" links are faster than "Local Download" links
# PDBBIND_GENERAL_URL: str = 'PDBbind_v2020_other_PL'
PDBBIND_GENERAL_URL = 'https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_other_PL.tar.gz'
if PDBBIND_GENERAL_URL is None:
    raise Exception("""PDBBIND_GENERAL_URL needs to be given, 
                    go to http://www.pdbbind.org.cn/download.php, 
                    and find the links to the general set""")
    
# PDBBIND_REFINED_URL: str = 'PDBbind_v2020_refined'
PDBBIND_REFINED_URL = 'https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz'
if PDBBIND_REFINED_URL is None:
    raise Exception("""PDBBIND_REFINED_URL needs to be given, 
                    go to http://www.pdbbind.org.cn/download.php, 
                    and find the links to the refined set""")

PDBBIND_DIRNAME = 'PDBbind'
PDBBIND_DIRPATH = os.path.join(ROOT_DIRPATH, 
                               PDBBIND_DIRNAME)
if not os.path.exists(PDBBIND_DIRPATH):
    os.mkdir(PDBBIND_DIRPATH)

PDBBIND_GENERAL_TARGZ_FILENAME = PDBBIND_GENERAL_URL.split('/')[-1]
PDBBIND_GENERAL_TARGZ_FILEPATH = os.path.join(PDBBIND_DIRPATH,
                                              PDBBIND_GENERAL_TARGZ_FILENAME)
PDBBIND_GENERAL_DIRNAME = 'general'
PDBBIND_GENERAL_DIRPATH = os.path.join(PDBBIND_DIRPATH,
                                       PDBBIND_GENERAL_DIRNAME)
# if not os.path.exists(PDBBIND_GENERAL_DIRPATH):
#     os.mkdir(PDBBIND_GENERAL_DIRPATH)

PDBBIND_REFINED_TARGZ_FILENAME = PDBBIND_REFINED_URL.split('/')[-1]
PDBBIND_REFINED_TARGZ_FILEPATH = os.path.join(PDBBIND_DIRPATH,
                                              PDBBIND_REFINED_TARGZ_FILENAME)
PDBBIND_REFINED_DIRNAME = 'refined'
PDBBIND_REFINED_DIRPATH = os.path.join(PDBBIND_DIRPATH,
                                       PDBBIND_REFINED_DIRNAME)
# if not os.path.exists(PDBBIND_REFINED_DIRPATH):
#     os.mkdir(PDBBIND_REFINED_DIRPATH)


CHEMBL_VERSION = 'chembl_29'
BASE_CHEMBL_URL = 'http://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases'
CHEMBL_FILENAME = f'{CHEMBL_VERSION}_sqlite'
CHEMBL_TARGZ_FILENAME = f'{CHEMBL_FILENAME}.tar.gz'
CHEMBL_URL = f'{BASE_CHEMBL_URL}/{CHEMBL_VERSION}/{CHEMBL_TARGZ_FILENAME}'
CHEMBL_DIRNAME = 'ChEMBL'
CHEMBL_DIRPATH = os.path.join(ROOT_DIRPATH, CHEMBL_DIRNAME)
if not os.path.exists(CHEMBL_DIRPATH):
    os.mkdir(CHEMBL_DIRPATH)
CHEMBL_SQLITE_PATH = os.path.join(CHEMBL_DIRPATH, 
                                  CHEMBL_FILENAME,
                                  f'{CHEMBL_VERSION}.db')
CHEMBL_TARGZ_FILEPATH = os.path.join(CHEMBL_DIRPATH,
                                     CHEMBL_TARGZ_FILENAME)

ENZYME_DAT_FILENAME = 'enzyme.dat'
BASE_ENZYME_URL = 'https://ftp.expasy.org/databases/enzyme'
ENZYME_URL = f'{BASE_ENZYME_URL}/{ENZYME_DAT_FILENAME}'
ENZYME_DIRNAME = 'ENZYME'
ENZYME_DIRPATH = os.path.join(ROOT_DIRPATH,
                              ENZYME_DIRNAME)
if not os.path.exists(ENZYME_DIRPATH):
    os.mkdir(ENZYME_DIRPATH)
ENZYME_DAT_FILEPATH = os.path.join(ENZYME_DIRPATH,
                                   ENZYME_DAT_FILENAME)

LIGANDEXPO_FILENAME = 'Components-smiles-stereo-cactvs.smi'
BASE_LIGANDEXPO_URL = 'http://ligand-expo.rcsb.org/dictionaries'
LIGANDEXPO_URL = f"{BASE_LIGANDEXPO_URL}/{LIGANDEXPO_FILENAME}"
LIGANDEXPO_DIRNAME = 'LigandExpo'
LIGANDEXPO_DIRPATH = os.path.join(ROOT_DIRPATH,
                                  LIGANDEXPO_DIRNAME)
if not os.path.exists(LIGANDEXPO_DIRPATH):
    os.mkdir(LIGANDEXPO_DIRPATH)
LIGANDEXPO_FILEPATH = os.path.join(LIGANDEXPO_DIRPATH,
                                   LIGANDEXPO_FILENAME)

WORKING_DIRNAME = 'conf_ranking'
WORKING_DIRPATH = os.path.join(ROOT_DIRPATH,
                               WORKING_DIRNAME)
if not os.path.exists(WORKING_DIRPATH):
    os.mkdir(WORKING_DIRPATH)

# All data relative to the current project are saved in DATA_DIRPATH
DATA_DIRPATH = os.path.join(WORKING_DIRPATH,
                            'data')
if not os.path.exists(DATA_DIRPATH):
    os.mkdir(DATA_DIRPATH)

BIO_CONF_DIRNAME = 'pdb_conf_ensembles'
# All bioactive conformation data are stored in BIO_CONF_DIRPATH
BIO_CONF_DIRPATH = os.path.join(DATA_DIRPATH,
                                BIO_CONF_DIRNAME)

GEN_CONF_DIRNAME = 'gen_conf_ensembles'
# All generated conformers for ligands in BIO_CONF_DIRPATH are stored in GEN_CONF_DIRPATH
GEN_CONF_DIRPATH = os.path.join(DATA_DIRPATH,
                                GEN_CONF_DIRNAME)

RMSD_DIRNAME = 'rmsds'
# All precomputed RMSD between the bioactive conformations and generated conformers 
# of each ligand are stored in RMSD_DIRPATH
RMSD_DIRPATH = os.path.join(DATA_DIRPATH,
                            RMSD_DIRNAME)

SPLITS_DIRNAME = 'splits'
# All information about the splits (i.e. which ligand SMILES or complex)
# are in train/val/test are stored in SPLITS_DIRPATH
SPLITS_DIRPATH = os.path.join(DATA_DIRPATH,
                              SPLITS_DIRNAME)

GOLD_PDBBIND_DIRNAME = 'gold_docking_pdbbind'
GOLD_PDBBIND_DIRPATH = os.path.join(WORKING_DIRPATH,
                                    GOLD_PDBBIND_DIRNAME)

RESULTS_DIRNAME = 'results'
RESULTS_DIRPATH = os.path.join(WORKING_DIRPATH,
                               RESULTS_DIRNAME)
if not os.path.exists(RESULTS_DIRPATH):
    os.mkdir(RESULTS_DIRPATH)

LOG_DIRNAME = 'lightning_logs'
LOG_DIRPATH = os.path.join(DATA_DIRPATH,
                           LOG_DIRNAME)

SCHNET_MODEL_NAME = 'SchNetModel'
SCHNET_CONFIG = {"num_interactions": 6,
                "cutoff": 10,
                "lr": 1e-5,
                'batch_size': 256}

DIMENET_MODEL_NAME = 'DimeNetModel'
DIMENET_CONFIG = {'hidden_channels': 128,
                    'out_channels': 1,
                    'num_blocks': 4,
                    'int_emb_size': 64,
                    'basis_emb_size': 8,
                    'out_emb_channels': 256,
                    'num_spherical': 7,
                    'num_radial':6 ,
                  "lr":1e-4,
                  'batch_size': 256}

COMENET_MODEL_NAME = 'ComENetModel'
COMENET_CONFIG = {"lr":1e-4,
                  'batch_size': 256}