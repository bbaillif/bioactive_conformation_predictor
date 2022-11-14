from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

class MolStandardizer() :

    def __init__(self) -> None:
        self.standardizer = Standardizer()
        
    def standardize(self,
                    mol: Mol,
                    neutralize: bool = True) -> Mol:
        new_mol = Mol(mol)
        standard_mol = self.standardizer.standardize(new_mol)
        
        # Uncharge for later comparison, because some functional groups might
        # be protonated. PDBBind stores a neutral pH protonated version of 
        # the ligand
        if neutralize:
            standard_mol = self.neutralize_mol(standard_mol)
        return standard_mol
    
    def neutralize_mol(self, 
                       mol: Mol) -> Mol:
        # see https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
        neutral_mol = Mol(mol)
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = neutral_mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = neutral_mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return neutral_mol
        
       