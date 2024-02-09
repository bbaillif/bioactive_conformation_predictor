from collections import defaultdict
from typing import List, Iterator
from rdkit import Chem
from rdkit.Chem import Mol


class PoseReader() :
    
    def __init__(self, 
                 file_path: str) -> None:
        """Class to read docked poses from GOLD in sdf or mol2 (prefered) formats

        :param file_path: Path of the sdf or mol2 file
        :type file_path: str
        """
        
        self.file_path = file_path
        self.file_format = file_path.split('.')[-1]
        assert self.file_format in ['mol2', 'sdf']
        
        if self.file_format == 'mol2' :
            self.mols = self.mols_from_mol2_file(self.file_path)
        elif self.file_format == 'sdf' :
            self.mols = self.mols_from_sdf(self.file_path)
        
    def mol2_block_to_mol(self,
                          mol2_block: List[str]) -> Mol:
        """Processes a mol2 block into a molecule (we cannot use the RDKit
        equivalent function because it does not process some of the GOLD
        properties stored in the file)

        :param mol2_block: Mol2 block: list of lines
        :type mol2_block: List[str]
        :return: Molecule
        :rtype: Mol
        """
        props = defaultdict(list)
        write_prop = False
        for line in mol2_block :
            if line.startswith('>') :
                write_prop = True
                prop_name = line.replace('<', '').replace('>', '').strip()
            elif write_prop :
                if line.strip() == '' :
                    props[prop_name] = ''.join(props[prop_name]) #.strip()
                    write_prop = False
                else : 
                    props[prop_name].append(line)
                    
        mol2_block = ''.join(mol2_block)
        mol = Chem.MolFromMol2Block(mol2_block)
        for prop_name, prop_value in props.items() :
            mol.SetProp(prop_name, prop_value)
            
        return mol

    def mols_from_mol2_file(self, 
                            file_path: str) -> List[Mol]:
        """Extract multiple molecules from the mol2 file (output mol2 file from 
        GOLD contains multiple docked poses)

        :param file_path: path to the mol2 file 
        :type file_path: str
        :return: List of molecules
        :rtype: List[Mol]
        """
        with open(file_path, 'r') as f: 
            lines = f.readlines()
        mols = []
        write_mol2_block = False
        mol2_block = []
        for line in lines[1:] : # avoid first line
            if line.startswith('@<TRIPOS>MOLECULE') :
                write_mol2_block = True
            if line.startswith('#       Name:') :
                write_mol2_block = False
                mol = self.mol2_block_to_mol(mol2_block)
                mols.append(mol)
                mol2_block = []
            write_current_line = True
    #         if '****' in line :
    #             write_current_line = False
            if write_mol2_block and write_current_line :
                mol2_block.append(line)
        mol = self.mol2_block_to_mol(mol2_block)
        mols.append(mol)
        #mol2_block = []
        return mols
    
    def mols_from_sdf(self,
                      file_path: str) -> List[Mol]:
        """Extract molecules from the sdf file

        :param file_path: Path to sdf file
        :type file_path: str
        :return: List of molecules
        :rtype: List[Mol]
        """
        sd_supplier = Chem.SDMolSupplier(file_path)
        mols = [mol for mol in sd_supplier]
        return mols
    
    def __iter__(self) -> Iterator:
       ''' Returns the Iterator object '''
       return iter(self.mols)