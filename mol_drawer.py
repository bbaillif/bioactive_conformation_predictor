import copy
import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol, Conformer
from rdkit.Chem.rdDepictor import Compute2DCoords
from collections.abc import Sequence
from typing import Dict

class MolDrawer() :
    
    def plot_values_for_mol(self, 
                             mol: Mol,
                             values: Sequence,
                            suffix: str='',
                            save_dir: str='test_mol') :
        mol_copy = copy.deepcopy(mol)
        # values = list(values)
        drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
        # drawer.drawOptions().addAtomIndices=True
        
        Compute2DCoords(mol_copy)
        for i, at in enumerate(mol_copy.GetAtoms()) :
            value = str(round(float(values[i]), 2))
            at.SetProp('atomNote', value)
        
        atom_ids = tuple(range(mol_copy.GetNumAtoms()))
        abs_values = np.abs(values)
        max_value = float(abs_values.max())
        colors = {i : self.score_to_rgba_color(values[i], -max_value, max_value) for i in atom_ids}
        
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, 
                                        mol_copy, 
                                        highlightAtoms=atom_ids, 
                                        highlightAtomColors=colors)
        drawer.FinishDrawing()
        
        if not os.path.exists(save_dir) :
            os.mkdir(save_dir)
        filepath = os.path.join(save_dir, 
                                f'mol_{suffix}.png')
        drawer.WriteDrawingText(filepath)
        
    def score_to_rgba_color(self, score, vmin=0, vmax=1) :
        cmap = cm.RdBu
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        normalized_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba = normalized_map.to_rgba(score)
        return rgba
        