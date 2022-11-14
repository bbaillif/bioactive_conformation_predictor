import os
import random

from tqdm import tqdm
from Bio import PDB
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool
from data.utils import PDBbindMetadataProcessor
from conf_ensemble import ConfEnsembleLibrary
from .protein_split import ProteinSplit

class UclusterSplit(ProteinSplit) :
    
    def __init__(self, 
                 split_type: str = 'ucluster', 
                 split_i: int = 0, 
                 cel_name: str = 'pdb_conf_ensembles', 
                 root: str = '/home/bb596/hdd/pdbbind_bioactive/data/', 
                 splits_dirname: str = 'splits', 
                 rmsd_name: str = 'rmsds',
                 clustering_dirname: str = 'protein_clustering') -> None:
        # We need to define the clustering parameters before the DataSplit init
        # Because the init uses the split_dataset function which requires
        # clustering dir
        self.clustering_dirname = clustering_dirname
        self.clustering_dirpath = os.path.join(root,
                                               clustering_dirname)
        if not os.path.exists(self.clustering_dirpath) :
            os.mkdir(self.clustering_dirpath)
        super().__init__(split_type, 
                         split_i, 
                         cel_name, 
                         root, 
                         splits_dirname, 
                         rmsd_name)
        
    
    def split_dataset(self,
                      frac_train = 0.8,
                      frac_val = 0.1):
        random.seed(42)
        
        fasta_filename = 'pdbbind_cel.fasta'
        fasta_filepath = os.path.join(self.clustering_dirpath, 
                                      fasta_filename)
        
        if not os.path.exists(fasta_filepath):
            print('Reading library')
            cel = ConfEnsembleLibrary()
            pdb_ids = []
            for name, ce in cel.library.items():
                for conf in ce.mol.GetConformers():
                    pdb_ids.append(conf.GetProp('PDB_ID'))
                    
            seqs = []
            with Pool() as p:
                iter_seqs = p.imap(self.get_seq_from_pdb_id, pdb_ids)
                for i, seq in tqdm(enumerate(iter_seqs), total=len(pdb_ids)):
                    seqs.append(seq)
                    
            print(f'Generate fasta file in {fasta_filepath}')
            with open(fasta_filepath, 'w') as f:
                for code, seq in zip(pdb_ids, seqs):
                    seq_record = SeqRecord(seq, 
                                        id=code, 
                                        description='')
                    SeqIO.write(seq_record, f, 'fasta')
            
        uc_filename = 'pdbbind_cel.uc'
        uc_filepath = os.path.join(self.clustering_dirpath, 
                                   uc_filename)
        
        if not os.path.exists(uc_filepath):
            import pdb;pdb.set_trace()
            print(f'Running ucluster, saving to {uc_filepath}')
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

        n_pdbs = len(pdb_to_cluster)
        train_cutoff = int(frac_train * n_pdbs)
        val_cutoff = int((frac_train + frac_val) * n_pdbs)

        clusters = [cluster for pdb_id, cluster in pdb_to_cluster.items()]
        unique_cluster_ids = list(set(clusters))

        protein_similarity_splits_dir_path = os.path.join(self.splits_dir_path, 
                                                          self.split_type)
        if not os.path.exists(protein_similarity_splits_dir_path) :
            os.mkdir(protein_similarity_splits_dir_path)
            
        for i in range(5) :
            
            current_split_dir_path = os.path.join(protein_similarity_splits_dir_path, 
                                                  str(i))
            if not os.path.exists(current_split_dir_path):
                os.mkdir(current_split_dir_path)
            
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
            
            with open(os.path.join(current_split_dir_path, f'train_pdbs.txt'), 'w') as f :
                for pdb in train_pdbs :
                    f.write(pdb)
                    f.write('\n')
                
            with open(os.path.join(current_split_dir_path, f'val_pdbs.txt'), 'w') as f :
                for pdb in val_pdbs :
                    f.write(pdb)
                    f.write('\n')
                
            with open(os.path.join(current_split_dir_path, f'test_pdbs.txt'), 'w') as f :
                for pdb in test_pdbs :
                    f.write(pdb)
                    f.write('\n')
                
                
    @staticmethod
    def get_seq_from_pdb_id(pdb_id):
        pmp = PDBbindMetadataProcessor(root='/home/bb596/hdd/PDBBind/')
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