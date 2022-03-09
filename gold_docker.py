import os
import time

from rdkit import Chem
from ccdc.docking import Docker
from ccdc.io import MoleculeReader
from ccdc_rdkit_connector import CcdcRdkitConnector
from ccdc.io import Entry, Molecule, MoleculeWriter
from ccdc.protein import Protein
from multiprocessing import Pool
from abc import abstractmethod

class FailedGOLDDockingException(Exception) :
    """Raised if GOLD does not return 0
    
    """
    def __init__(self):
        message = 'GOLD did not return 0, docking failed'
        super().__init__(message)

class GOLDDocker() :
    """Initialize a GOLD Docker with a protein, the native ligand and 
    default settings
        
    :param protein_path: Path to the protein pdb file
    :type protein_path: str
    :param native_ligand_path: Path to the ligand mol2 file
    :type native_ligand_path: str
    :param output_dir: Path where to store params and results
    :type output_dir: str
    """
    
    def __init__(self,
                 protein_path: str,
                 native_ligand_path: str,
                 experiment_id: str,
                 output_dir: str='gold_docking_dude',
                 prepare_protein: bool=False
                 ):
        
        self.protein_path = protein_path
        self.native_ligand_path = native_ligand_path
        self.experiment_id = experiment_id
        self.output_dir = os.path.abspath(output_dir)
        self.prepare_protein = prepare_protein
        
        if not os.path.exists(self.output_dir) :
            os.mkdir(self.output_dir)
        
        self.docker = Docker()
        
        # Setup some default settings for our experiments
        self.settings = self.docker.settings
        self.settings.fitness_function = 'plp'
        self.settings.autoscale = 10.
        self.settings.early_termination = False
        self.settings.diverse_solutions = True
        
        self.settings.write_options= ['NO_FIT_PTS_FILES']

        self.ligand_preparation = Docker.LigandPreparation()
        
        self.ccdc_rdkit_connector = CcdcRdkitConnector()
        
        if self.prepare_protein :
            
            initial_protein_filename = os.path.basename(protein_path)
            pdb_id = initial_protein_filename.split('.pdb')[0]
            new_protein_filename = f'{pdb_id}_prepared.pdb'
            protein_dir = os.path.dirname(protein_path)
            prepared_protein_path = os.path.join(protein_dir, 
                                                 new_protein_filename)
            
            if not os.path.exists(prepared_protein_path) :
                protein = Protein.from_file(protein_path)
                protein.add_hydrogens()
                for ligand in protein.ligands :
                    protein.remove_ligand(ligand.identifier)
                protein.remove_all_metals()
                protein.remove_all_waters()
                
                with MoleculeWriter(prepared_protein_path) as protein_writer :
                    protein_writer.write(protein)
                    
            self.settings.add_protein_file(prepared_protein_path)
            
        else :
            self.settings.add_protein_file(protein_path)
                
        self.settings.reference_ligand_file = native_ligand_path
        
        docker_output_dir = os.path.join(self.output_dir, 
                                         self.experiment_id)
        if not os.path.exists(docker_output_dir) :
            os.mkdir(docker_output_dir)
        self.settings.output_directory = docker_output_dir

        # Define binding site
        self.protein = self.settings.proteins[0]
        self.native_ligand = MoleculeReader(self.native_ligand_path)[0]
        bs = self.settings.BindingSiteFromLigand(protein=self.protein, 
                                                 ligand=self.native_ligand)
        self.settings.binding_site = bs
            
        self.docked_ligand_name = 'docked_ligands.mol2'

    def dock_molecule(self, 
                      ccdc_mol: Molecule,
                      mol_id: str,
                      n_poses: int=20,
                      rigid: bool=False,
                      return_runtime: bool=False,):
        """Dock a single molecule using GOLD 
        
        :param ccdc_mol: Molecule to dock
        :type ccdc_mol: ccdc.io.Molecule
        :param mol_id: identifier to give to the molecule (for results 
            directory)
        :type mol_id: str
        :param n_poses: Number of output poses of docking
        :type n_poses: int
        :param rigid: Fix the torsions to perform rigid ligand docking
        :type rigid: bool
        :param return_runtime: Whether or not to return docking runtime
        :type return_runtime: bool
        :return: docking results and runtime if asked
        :rtype: DockingResults (and float if return_runtime)
        """
        return self.dock_molecules(ccdc_mols=[ccdc_mol],
                                    mol_id=mol_id,
                                    n_poses=n_poses,
                                    rigid=rigid,
                                    return_runtime=return_runtime)

    def dock_molecules(self, 
                      ccdc_mols: Molecule,
                      mol_id: str,
                      n_poses: int=20,
                      rigid: bool=False,
                      return_runtime: bool=False,
                      ):
        """Dock molecules using GOLD 
        
        :param ccdc_mols: Molecules to dock
        :type ccdc_mols: list[ccdc.io.Molecule]
        :param mol_id: identifier to give to the molecule (for results 
            directory)
        :type mol_id: str
        :param n_poses: Number of output poses of docking
        :type n_poses: int
        :param rigid: Fix the torsions to perform rigid ligand docking
        :type rigid: bool
        :param return_runtime: Whether or not to return docking runtime
        :type return_runtime: bool
        :return: docking results and runtime if asked
        :rtype: DockingResults (and float if return_runtime)
        """
        
        self.settings.clear_ligand_files() # avoid docking previous ligands
        
        if rigid :
            self.settings.fix_ligand_rotatable_bonds = 'all'
        else :
            self.settings.fix_ligand_rotatable_bonds = None
        
        mol_output_dir = os.path.join(self.output_dir,
                                      self.experiment_id,
                                      mol_id)
        if not os.path.exists(mol_output_dir) :
            os.mkdir(mol_output_dir)
        self.settings.output_directory = mol_output_dir
        
        poses_output_file = os.path.join(mol_output_dir, 
                                         self.docked_ligand_name)
        self.settings.output_file = poses_output_file

        for conf_id, ccdc_mol in enumerate(ccdc_mols) :
            ligand_file = os.path.join(mol_output_dir, 
                                                f'ligand_{conf_id}.mol2')
            self.prepare_ligand(ccdc_mol=ccdc_mol,
                                ligand_file=ligand_file,
                                n_poses=n_poses)
        
        start_time = time.time()
        conf_file_name = os.path.join(mol_output_dir, f'api_gold.conf')
        results = self.docker.dock(file_name=conf_file_name)
        if results.return_code :
            raise FailedGOLDDockingException()
        runtime = time.time() - start_time
        
        if return_runtime :
            return results, runtime
        else :
            return results

    def prepare_ligand(self,
                       ccdc_mol: Molecule,
                       ligand_file: str,
                       n_poses: int,
                       ):
        
        ligand = self.ligand_preparation.prepare(Entry.from_molecule(ccdc_mol))
        mol2_string = ligand.to_string(format='mol2')
        with open(ligand_file, 'w') as writer :
            writer.write(mol2_string)
        self.settings.add_ligand_file(ligand_file, n_poses)
