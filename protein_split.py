import os 
import random

from tqdm import tqdm
from Bio import PDB
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool
from pdbbind_metadata_processor import PDBBindMetadataProcessor
from conf_ensemble import ConfEnsembleLibrary

def get_seq_from_pdb_id(pdb_id):
    pmp = PDBBindMetadataProcessor(root='/home/bb596/hdd/PDBBind/')
    protein_path, ligand_pathes = pmp.get_pdb_id_pathes(pdb_id)
    protein_pdb = protein_path
    pocket_pdb = protein_path.replace('protein', 'pocket')
    parser = PDB.PDBParser(QUIET=True)
    pocket = parser.get_structure(pdb_id, pocket_pdb)
    protein = parser.get_structure(pdb_id, protein_pdb)
    
    longest_chain = None
    for chain in pocket.get_chains():
        if chain.id == ' ': 
            continue
        if longest_chain is None or len(chain) > len(longest_chain):
            longest_chain = chain

    ppb = PDB.PPBuilder()
    for chain in protein.get_chains():
        if chain.id == longest_chain.id:
            seqs = [pp.get_sequence() for pp in ppb.build_peptides(chain)]
            seq_str = ''.join([str(seq) for seq in seqs])
            return Seq(seq_str)

root = '/home/bb596/hdd/pdbbind_bioactive/data'
protein_dir = os.path.join(root, 'protein_clustering')
if not os.path.exists(protein_dir) :
    os.mkdir(protein_dir)

print('Reading library')
cel = ConfEnsembleLibrary()
pdb_ids = []
for name, ce in cel.library.items():
    for conf in ce.mol.GetConformers():
        pdb_ids.append(conf.GetProp('PDB_ID'))

codes = pdb_ids
seqs = []
with Pool() as p:
    iter_seqs = p.imap(get_seq_from_pdb_id, pdb_ids)
    for i, seq in tqdm(enumerate(iter_seqs), total=len(pdb_ids)):
        seqs.append(seq)

fasta_filename = 'pdbbind_cel.fasta'
fasta_filepath = os.path.join(protein_dir, fasta_filename)
with open(fasta_filepath, 'w') as f:
    for code, seq in zip(pdb_ids, seqs):
        SeqIO.write(SeqRecord(seq, id=code, description=''), f, 'fasta')
    
uc_filename = 'pdbbind_cel.uc'
uc_filepath = os.path.join(protein_dir, uc_filename)
    
# run the UCLUST algorithm
command = f'./usearch11.0.667_i86linux32 -cluster_fast {fasta_filepath} -id 0.4 -uc {uc_filepath}'
os.system(command)

pdb_to_cluster = {}
with open(uc_filepath) as f:
    for line in f:
        fields = line.split()
        cluster = fields[1]
        pdb_id = fields[8]
        pdb_to_cluster[pdb_id] = cluster
        
frac_train = 0.8
frac_val = 0.1

train_cutoff = int(frac_train * len(pdb_to_cluster))
val_cutoff = int((frac_train + frac_val) * len(pdb_to_cluster))
train_inds = []
val_inds = []
test_inds = []

clusters = [cluster for pdb_id, cluster in pdb_to_cluster.items()]
unique_cluster_ids = list(set(clusters))

protein_similarity_splits_dir_name = 'protein_splits'
protein_similarity_splits_dir_path = os.path.join(root, protein_similarity_splits_dir_name)
if not os.path.exists(protein_similarity_splits_dir_path) :
    os.mkdir(protein_similarity_splits_dir_path)
    
for i in range(5) :
    
    random.shuffle(unique_cluster_ids)
    
    train_pdbs = []
    val_pdbs = []
    test_pdbs = []
    
    for current_cluster_id in unique_cluster_ids:
        pdbs = [pdb_id 
                for pdb_id, cluster_id in pdb_to_cluster.items() 
                if cluster_id == current_cluster_id]
        if len(train_pdbs) + len(pdbs) > train_cutoff:
            if len(train_pdbs) + len(val_pdbs) + len(pdbs) > val_cutoff:
                test_pdbs += pdbs
            else:
                val_pdbs += pdbs
        else:
            train_pdbs += pdbs
    
    with open(os.path.join(protein_similarity_splits_dir_path, f'train_pdb_{i}.txt'), 'w') as f :
        for pdb in train_pdbs :
            f.write(pdb)
            f.write('\n')
        
    with open(os.path.join(protein_similarity_splits_dir_path, f'val_pdb_{i}.txt'), 'w') as f :
        for pdb in val_pdbs :
            f.write(pdb)
            f.write('\n')
        
    with open(os.path.join(protein_similarity_splits_dir_path, f'test_pdb_{i}.txt'), 'w') as f :
        for pdb in test_pdbs :
            f.write(pdb)
            f.write('\n')