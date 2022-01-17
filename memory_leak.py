import time
import copy
import torch
from tqdm import tqdm
from conf_ensemble_dataset_in_memory import ConfEnsembleDataset

for chunk_number in tqdm(range(2)) :
    data, slices = torch.load(f'data/processed/platinum_dataset_{chunk_number}.pt')

# del data
# del slices
time.sleep(1000)