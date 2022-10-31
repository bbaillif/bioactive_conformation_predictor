import argparse
import os 

from tqdm import tqdm
from pathlib import Path
from datetime import datetime as dt
from Bio import PDB
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool
from pdbbind_metadata_processor import PDBBindMetadataProcessor
from conf_ensemble import ConfEnsembleLibrary

start = dt.now()

def SeqFromPDBCode(code):
    pmp = PDBBindMetadataProcessor(root='/home/bb596/hdd/PDBBind/')
    protein_path, ligand_pathes = pmp.get_pdb_id_pathes(pdb_id=code)
    protein_pdb = protein_path
    pocket_pdb = protein_path.replace('protein', 'pocket')
    parser = PDB.PDBParser(QUIET=True)
    try:
        protein = parser.get_structure(code, protein_pdb)
    except:
        print('fail to read {}'.format(code))
        return None
    ppb = PDB.PPBuilder()
    seqs = []
    for chain in protein.get_chains():
        seqs.extend([i.get_sequence() for i in ppb.build_peptides(chain)])
        seq_str = ''.join([str(i) for i in seqs])
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

fasta_file = 'pdbbind_cel.fasta'
with open(fasta_file, 'w') as f:
    for code, seq in zip(codes, seqs):
        SeqIO.write(SeqRecord(seq, id=code, description=''), f, 'fasta')
print('Sequences save at {}\n'.format(fasta_file))
print('Elapsed time {}.'.format(dt.now() - start))