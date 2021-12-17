import ipywidgets
import py3Dmol
import unittest

from ipywidgets import interact, fixed, IntSlider
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from typing import Tuple


class MolViewer():
    
    def view(self, mol: Mol) :
        """View a molecule in 3D, with a slider to explore conformations
        Args:
        ----
            mol: rdMol, molecule to show
        
        """
        
        def conf_viewer(mol: Mol, conf_id: int):# -> py3Dmol.view:
            return self.molTo3DView(mol=mol, 
                                    conf_id=conf_id).show()

        interact(conf_viewer, 
                 mol=fixed(mol), 
                 conf_id=ipywidgets.IntSlider(min=0, max=mol.GetNumConformers() - 1, step=1))
        
    def molTo3DView(self, 
                    mol: Mol, 
                    conf_id: int = -1, 
                    size: Tuple[int, int] = (300, 300), 
                    style: str = "stick", 
                    surface: bool = False, 
                    opacity: float = 0.5):# -> py3Dmol.view:
        """Draw molecule in 3D
    
        Args:
        ----
            mol: rdMol, molecule to show
            size: tuple(int, int), canvas size
            style: str, type of drawing molecule
                   style can be 'line', 'stick', 'sphere', 'carton'
            surface, bool, display SAS
            opacity, float, opacity of surface, range 0.0-1.0
        Return:
        ----
            viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
        """
        
        assert style in ('line', 'stick', 'sphere', 'carton')
        
        mblock = Chem.MolToMolBlock(mol, confId=conf_id)
        viewer = py3Dmol.view(width=size[0], height=size[1])
        viewer.addModel(mblock, 'mol')
        viewer.setStyle({style:{}})
        if surface:
            viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
        viewer.zoomTo()
        return viewer
    
    
class MolViewerTest(unittest.TestCase):
    
    def test_view(self):
        mol = Chem.MolFromSmiles('c1ccccc1')
        EmbedMultipleConfs(mol)
        MolViewer().view(mol)