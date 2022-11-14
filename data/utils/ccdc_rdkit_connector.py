import copy

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
from ccdc.molecule import Molecule

class ConversionError(Exception) :
    pass

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
    
    def ccdc_ensemble_to_rdkit_mol(self, 
                                   ccdc_ensemble, 
                                   rdkit_mol=None, 
                                   generated=False,
                                   remove_input_conformers=False) :
        """Add ensemble to the given rdkit_mol, or to a new rdkit_mol
        
        :param ccdc_ensemble: Ensemble of subtypes of ccdc entries
        :type ccdc_emsemble: ConformerHitList or DockedLigand ensemble
        :param rdkit_mol: RDKit molecule to add conformations to (default is 
            None, in which case the rdkit_mol will be generated from the first
            element of the ensemble)
        :type rdkit_mol: rdkit.Chem.rdchem.Mol or None
        :returns: RDKit molecule and ids of the added conformations
        :rtype: rdkit.Chem.rdchem.Mol, List[int]
        """
        
        new_conf_ids = []

        if rdkit_mol is None :
            rdkit_mol = self.ccdc_mol_to_rdkit_mol(ccdc_ensemble[0].molecule)
            ccdc_ensemble = [entry 
                             for i, entry in enumerate(ccdc_ensemble) 
                             if i in range(1, len(ccdc_ensemble))]
            # because first mol is template
            # and for some reason docking results.ligands are not sliceable

            if rdkit_mol is None :
                raise ConversionError()

        new_rdkit_mol = copy.deepcopy(rdkit_mol)
        if remove_input_conformers :
            new_rdkit_mol.RemoveAllConformers()
        for entry in ccdc_ensemble :
            new_rdkit_conf = copy.deepcopy(rdkit_mol).GetConformer()
            for i in range(new_rdkit_conf.GetNumAtoms()) :
                atom = entry.molecule.atoms[i]
                point3d = Point3D(*atom.coordinates)
                new_rdkit_conf.SetAtomPosition(i, point3d)
            if generated :
                new_rdkit_conf.SetProp('Generator', 'CCDC')
            conf_id = new_rdkit_mol.AddConformer(new_rdkit_conf, assignId=True)
            new_conf_ids.append(conf_id)
        
        return new_rdkit_mol, new_conf_ids
    
    def ccdc_mol_to_rdkit_mol(self, ccdc_mol) :
        ccdc_mol.remove_atoms([atom 
                               for atom in ccdc_mol.atoms 
                               if atom.atomic_number < 1])
        mol2block = ccdc_mol.to_string()
        return Chem.MolFromMol2Block(mol2block, removeHs=False)
    
    def ccdc_mols_to_rdkit_mol_conformers(self, ccdc_mols, rdkit_mol) :
        """Add conformers to the rdkit_mol in place
        Args:
            ccdc_mols : List[Molecule]
            rdkit_mol
        Returns:
            generated_conf_ids : Id of the added conformations in the rdkit molecule
        """
        
        generated_conf_ids = []

        for ccdc_mol in ccdc_mols :
            new_rdkit_conf = copy.deepcopy(rdkit_mol).GetConformer()
            for i in range(new_rdkit_conf.GetNumAtoms()) :
                atom = ccdc_mol.atoms[i]
                point3d = Point3D(*atom.coordinates)
                new_rdkit_conf.SetAtomPosition(i, point3d)
            new_rdkit_conf.SetProp('Generator', 'CCDC')
            conf_id = rdkit_mol.AddConformer(new_rdkit_conf, assignId=True)
            generated_conf_ids.append(conf_id)