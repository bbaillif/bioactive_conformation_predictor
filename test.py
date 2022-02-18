from pdbbind_docking_preprocessing import PDBBindDocking
import os

pdbbind_docking = PDBBindDocking()
successful_pdb_ids = ['4uiu', '3sqq', '6bh0', '4jvb',
                      '4kn2', '4tkg', '3d8y', '5yz7', 
                      '3gnv', '2pg2', '6drz', '6ge0', 
                      '2o22', '2nns', '6oip', '1wbw', 
                      '5bqh', '2oh0', '5fqv', '2v3e', 
                      '3d67', '3ti8']    
pdbbind_docking.docking_analysis(pdb_ids=successful_pdb_ids)