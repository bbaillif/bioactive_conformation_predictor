from rdkit import Chem
from mol_standardizer import MolStandardizer

class PDBLigandExpo() :

    def __init__(self,
                 smiles_filepath = '/home/bb596/hdd/PDB/Components-smiles-stereo-cactvs.smi') :
        self.smiles_filepath = smiles_filepath
        self.smiles_d = self._get_smiles_d() # Links a PDB ligand name to corresponding smiles in cactvs
        self.mol_standardizer = MolStandardizer()
        
    def _get_smiles_d(self) :
        d = {}
        with open(self.smiles_filepath, 'r') as f :
            lines = f.readlines() # a line is SMILES\tLIGANDID\tLIGANDFULLNAME
        for line in lines :
            l = line.strip().split('\t')
            if len(l) == 3 : # there might be lines having no smiles
                smiles = l[0]
                ligand_name = l[1]
                d[ligand_name] = smiles
        return d
    
    def get_smiles(self,
                   ligand_name) :
        smiles = None
        if ligand_name in self.smiles_d : 
            smiles = self.smiles_d[ligand_name]
        else :
            raise Exception(f'Ligand name {ligand_name} not in Ligand expo')
        return smiles
    
    def get_standard_ligand(self,
                            ligand_name,
                            standardize=True) :
        smiles = self.get_smiles(ligand_name)
        mol = Chem.MolFromSmiles(smiles)
        if standardize:
            standard_mol = self.mol_standardizer.standardize(mol, neutralize=False)
        else:
            standard_mol = mol
        return standard_mol