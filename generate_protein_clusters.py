import json
import pandas as pd
import pickle

from tqdm import tqdm
from datetime import datetime as dt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from Bio import PDB
from Bio.Seq import Seq
from Bio import Align
from multiprocessing import Pool
from pdbbind_metadata_processor import PDBBindMetadataProcessor

def SeqFromPDBCode(code):
    protein_path, ligand_pathes = PDBBindMetadataProcessor().get_pdb_id_pathes(pdb_id=code)
    protein_pdb = protein_path
    pocket_pdb = protein_path.replace('protein', 'pocket')
    parser = PDB.PDBParser(QUIET=True)
    chain_id = None
    try:
        pocket = parser.get_structure(code, pocket_pdb)
        protein = parser.get_structure(code, protein_pdb)
    except:
        return None
    longest_chain = None
    for chain in pocket.get_chains():
        if chain.id == ' ': continue
        if longest_chain is None or len(chain) > len(longest_chain):
            longest_chain = chain
    if longest_chain is None:
        return None
    ppb = PDB.PPBuilder()
    for chain in protein.get_chains():
        if chain.id == longest_chain.id:
            seqs = [i.get_sequence() for i in ppb.build_peptides(chain)]
            seq_str = ''.join([str(i) for i in seqs])
            return Seq(seq_str)

class PairwiseSequenceSimilarity(object) :
    
    def __init__(self,
                seqs) :
        self.aligner = Align.PairwiseAligner()
        self.seqs = seqs
        
    def get_alignment(self,
                      seqA, 
                      seqB) :
        alignments = self.aligner.align(seqA, seqB)
        alignment = alignments[0]
        return alignment
        
    def get_identity(self,
                    alignment) :
        matches = alignment.score
        alignment_length = alignment.shape[1]
        identity = (matches / alignment_length) * 100
        return identity
        
    def get_distance(self, 
                     seqA, 
                     seqB) :
        alignment = self.get_alignment(seqA, seqB)
        identity = self.get_identity(alignment)
        distance = 100 - identity
        return 100 - identity
        
    def _get_distance(self, i, j) :
        i = int(i[0])
        j = int(j[0])
        seqA = self.seqs[i]
        seqB = self.seqs[j]
        return self.get_distance(seqA, seqB)
    
    def get_distance_matrix(self) :
        #start_time = time.time()
        seqs_2d = [[i] for i in range(len(self.seqs))]
        distance_matrix = pairwise_distances(seqs_2d, metric=self._get_distance, n_jobs=-1)
        #print(f'Time elapsed: {time.time() - start_time} seconds')
        return distance_matrix

start = dt.now()

smiles_df = pd.read_csv('data/smiles_df.csv')
pdb_ids = smiles_df[smiles_df['dataset'] == 'pdbbind']['id'].unique()
codes = pdb_ids

seqs = []
Nones = []
p = Pool()
iter_seqs = p.imap(SeqFromPDBCode, codes)
for i, seq in tqdm(enumerate(iter_seqs), total=len(codes)):
    if seq is None:
        Nones.append(i)
    else:
        seqs.append(seq)
p.close()
print('succeeded {}/{}\n'.format(len(seqs), len(codes)))
codes = [j for i, j in enumerate(codes) if i not in Nones]

code_file = 'codes.json'
with open(code_file, 'w') as f:
    json.dump(codes, f, indent=4)

seqs_2d = [[i] for i in range(len(seqs))]

psa = PairwiseSequenceSimilarity(seqs=seqs)
dm = psa.get_distance_matrix()

with open('protein_sequence_distance_matrix.p', 'wb') as f :
    pickle.dump(dm, f)

distances = squareform(dm)

Z = linkage(distances)

T = fcluster(Z, t=20, criterion='distance')
print("Found {} clusters with max 20% different.\n".format(max(T)+1))

cluster_file = 'protein_clusters.json'
with open(cluster_file, 'w') as f:
    json.dump(T.tolist(), f, indent=4)
print('Flat cluster result save at {}\n'.format(cluster_file))

pdb_to_cluster = {codes[i] : int(cluster_id) for i, cluster_id in enumerate(T)}
cluster_file = 'pdb_clusters.json'
with open(cluster_file, 'w') as f:
    json.dump(pdb_to_cluster, f, indent=4)
print('Flat cluster result save at {}\n'.format(cluster_file))

print('Elapsed time {}.'.format(dt.now() - start))