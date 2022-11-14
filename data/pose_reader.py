from rdkit import Chem
from collections import defaultdict

class PoseReader() :
    
    def __init__(self, 
                 file_path) :
        
        self.file_path = file_path
        self.file_format = file_path.split('.')[-1]
        assert self.file_format in ['mol2', 'sdf']
        
        if self.file_format == 'mol2' :
            self.mols = self.mols_from_mol2_file(self.file_path)
        elif self.file_format == 'sdf' :
            self.mols = self.mols_from_sdf(self.file_path)
        
    def mol2_block_to_mol(self, mol2_block) :
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

    def mols_from_mol2_file(self, file_path) :
        with open(file_path, 'r') as f: 
            lines = f.readlines()
        mols = []
        write_mol2_block = False
        mol2_block = []
        for line in lines[1:] : # avoir first line
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
                      file_path: str) :
        sd_supplier = Chem.SDMolSupplier(file_path)
        mols = [mol for mol in sd_supplier]
        return mols
    
    def __iter__(self):
       ''' Returns the Iterator object '''
       return iter(self.mols)