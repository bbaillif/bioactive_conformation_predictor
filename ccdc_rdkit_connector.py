import copy

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
from ccdc.molecule import Molecule

class CcdcRdkitConnector() :
    
    def rdkit_conf_to_ccdc_mol(self, rdkit_mol, conf_id=0) :
        """Create a ccdc molecule for a given conformation from a rdkit molecule
        Communication via mol block
        Args :
            rdkit_mol : Input molecule
            conf_id : Id of the conformation in the rdkit_mol
        Returns :
            ccdc molecule
        """
        molblock = Chem.MolToMolBlock(rdkit_mol, confId=conf_id)
        return Molecule.from_string(molblock)
    
    def ccdc_conformers_to_rdkit_mol(self, ccdc_conformers, rdkit_mol) :
        """Add conformers to the rdkit_mol in place
        Args:
            ccdc_conformers : ConformerHitList
            rdkit_mol
        Returns:
            generated_conf_ids : Id of the added conformations in the rdkit molecule
        """
        
        generated_conf_ids = []

        for conformer in ccdc_conformers :
            new_rdkit_conf = copy.deepcopy(rdkit_mol).GetConformer()
            for i in range(new_rdkit_conf.GetNumAtoms()) :
                atom = conformer.molecule.atoms[i]
                point3d = Point3D(*atom.coordinates)
                new_rdkit_conf.SetAtomPosition(i, point3d)
            new_rdkit_conf.SetProp('Generator', 'CCDC')
            conf_id = rdkit_mol.AddConformer(new_rdkit_conf, assignId=True)
            generated_conf_ids.append(conf_id)

        return generated_conf_ids