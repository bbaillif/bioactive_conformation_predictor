import numpy as np
import pandas as pd
import os

from .conf_ranker import ConfRanker
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, EditableMol
from typing import Sequence, Tuple, Union
from conf_ensemble import ConfEnsembleLibrary
from data.utils.similarity_search import SimilaritySearch
from data.utils.enzyme import ENZYME
from data.utils.pdbbind import PDBbind
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.TorsionFingerprints import GetTFDBetweenConformers
from params import PDBBIND_DIRPATH


class TFD2SimRefMCSRanker(ConfRanker):
    """For each test molecule, lookup the most similar molecule in the 
        training set then rank conformers using ascending TFD of the MCS to this
        closest molecule

    :param cel: Conf ensemble library of training molecules
    :type cel: ConfEnsembleLibrary
    :param name: Name of the ranker
    :type name: str
    :param ascending: Set to True to rank conformer by ascending
        values, or False for descending values, defaults to False
    :type ascending: bool, optional
    """
    
    def __init__(self,
                 cel: ConfEnsembleLibrary,
                 name: str = 'TFD2RefMCS',
                 ascending: bool = True) -> None:
        super().__init__(name=name, 
                         ascending=ascending)
        self.ascending = ascending
        self.cel = cel
        

        ec = ENZYME()
        enzyme_table = ec.get_table()

        pdb_df_path = os.path.join(self.cel.cel_dir, 'pdb_df.csv')
        pdb_df = pd.read_csv(pdb_df_path)
        cel_pdb_df = self.cel.cel_df.merge(pdb_df, 
                                           left_on='ensemble_name', 
                                           right_on='ligand_name', 
                                           how='left')
        pdbbind_proc = PDBbind(remove_mers=True,
                                remove_unknown_ligand_name=True,
                                remove_unknown_uniprot=True)
        pdbbind_df = pdbbind_proc.get_master_dataframe()
        merged_df = cel_pdb_df.merge(pdbbind_df, 
                                     left_on='pdb_id', 
                                     right_on='PDB code',
                                     how='left')
        self.merged_df = merged_df.merge(enzyme_table, 
                                         left_on='Uniprot ID', 
                                         right_on='uniprot_id',
                                         how='left')
        
        smiles_list = self.cel.cel_df['smiles'].unique()
        self.similarity_search = SimilaritySearch(smiles_list=smiles_list)
        print('TFD ranker ready, training compounds are loaded')


    def get_input_list_for_mol(self,
                               mol: Mol,
                               return_mcs: bool = True
                               ) -> Union[Sequence[float],
                                          Tuple[Sequence[float], Sequence[str]]]:
        """Get minimum TFDs for a molecule

        :param mol: Input molecule
        :type mol: Mol
        :param return_mcs: Set to True to return the MCS to reference molecule, 
            defaults to True
        :type return_mcs: bool, optional
        :return: Minimum TFD, and if return_mcs also returns the MCS
        :rtype: Union[Sequence[float], Tuple[Sequence[float], Sequence[str]]]
        """
        ref_mol = self.find_closest_mols_in_cel(mol)
        min_tfds, mcs_smarts = self.get_tfds_to_ref_mcs(mol, ref_mol)
        
        if return_mcs:
            return min_tfds, mcs_smarts
        else:
            return min_tfds


    def find_closest_mols_in_cel(self, 
                                mol: Mol) -> Mol:
        """Find the closest mol in the reference conf ensemble library

        :param mol: Input molecule
        :type mol: Mol
        :return: Closest molecule
        :rtype: Mol
        """
        smiles = Chem.MolToSmiles(mol)
        closest_smiles, sims = self.similarity_search.find_closest_in_set(smiles)
        cel_df = self.cel.cel_df
        ensemble_name = cel_df[cel_df['smiles'] == closest_smiles[0]]['ensemble_name'].values[0]
        conf_ensemble = self.cel.library[ensemble_name]
        return conf_ensemble.mol


    def get_tfds_to_ref_mcs(self,
                           mol: Mol,
                           ref_mol: Mol,
                           enzyme_classes: str = None
                           ) -> Tuple[Sequence[float], Sequence[str]]:
        """Get the TFDs of the MCS to the reference molecule

        :param mol: Input molecule
        :type mol: Mol
        :param ref_mol: Reference molecule
        :type ref_mol: Mol
        :param enzyme_classes: Restrict the reference search to some enzyme classes, 
            defaults to None
        :type enzyme_classes: str, optional
        :return: TFDs and MCSs
        :rtype: Tuple[Sequence[float], Sequence[str]]
        """
        mcs_smarts = None
        try:
            mcs = FindMCS([ref_mol, mol], 
                          timeout=10, 
                          matchChiralTag=True,)
                        #   ringMatchesRingOnly=True)
            mcs_smarts = mcs.smartsString
            mcs_mol = Chem.MolFromSmarts(mcs_smarts)
            
            ref_mol_match = ref_mol.GetSubstructMatch(mcs_mol)
            pdb_edit_mol = self.get_editable_mol_match(ref_mol, ref_mol_match)
            new_ref_mol = pdb_edit_mol.GetMol()
            new_pdb_match = new_ref_mol.GetSubstructMatch(mcs_mol)
            new_ref_mol = Chem.RenumberAtoms(new_ref_mol, new_pdb_match)
            
            mol_match = mol.GetSubstructMatch(mcs_mol)
            gen_edit_mol = self.get_editable_mol_match(mol, mol_match)
            new_mol = gen_edit_mol.GetMol()
            new_gen_match = new_mol.GetSubstructMatch(mcs_mol)
            new_mol = Chem.RenumberAtoms(new_mol, new_gen_match)
            
            bio_conf_ids = []
            for conf in new_ref_mol.GetConformers():
                add_conf = False
                if enzyme_classes is not None:
                    if conf.HasProp('PDB_ID'):
                        pdb_id = conf.GetProp('PDB_ID')
                        subset_df = self.merged_df[self.merged_df['PDB code'] == pdb_id]
                        if enzyme_classes in subset_df['level_4'].unique():
                            add_conf = True
                else:
                    add_conf = True
                    
                if add_conf:
                    conf_id = mcs_mol.AddConformer(conf, assignId=True)
                    bio_conf_ids.append(conf_id)
                
            if len(bio_conf_ids) == 0:
                print('No enzyme class match, taking all known conformations')
                for conf in new_ref_mol.GetConformers():
                    conf_id = mcs_mol.AddConformer(conf, assignId=True)
                    bio_conf_ids.append(conf_id)
                
            gen_conf_ids = []
            for conf in new_mol.GetConformers():
                conf_id = mcs_mol.AddConformer(conf, assignId=True)
                gen_conf_ids.append(conf_id)
                
            Chem.SanitizeMol(mcs_mol)
            # TODO: can be optimized by computing only TFD between bio_conf and gen_conf
            tfds = []
            for bio_conf_id in bio_conf_ids:
                current_tfds = GetTFDBetweenConformers(mcs_mol, 
                                                        confIds1=[bio_conf_id], 
                                                        confIds2=gen_conf_ids)
                tfds.append(current_tfds)
            tfds = np.array(tfds)
            # tfd_matrix = GetTFDMatrix(mcs_mol)
            # tfd_matrix = self.get_full_matrix_from_tril(tfd_matrix, 
            #                                             n=mcs_mol.GetNumConformers())
            # n_ref_confs = len(bio_conf_idx)
            # tfds = tfd_matrix[:n_ref_confs, n_ref_confs:]
            min_tfds = tfds.min(0)
            assert len(min_tfds) == mol.GetNumConformers(), 'N must match'
            
        except IndexError:
            min_tfds = np.ones(mol.GetNumConformers())
            
        except Exception as e:
            print(e)
            min_tfds = np.ones(mol.GetNumConformers())
            
        return min_tfds, mcs_smarts


    @staticmethod
    def get_editable_mol_match(mol: Mol, 
                               match: Sequence[int]
                               ) -> EditableMol:
        """Get editable mol corresponding to a list of atom indexes in match

        :param mol: Input molecule
        :type mol: Mol
        :param match: List of atoms to match
        :type match: Sequence[int]
        :return: EditableMol containing only atom in match
        :rtype: EditableMol
        """
        edit_mol = EditableMol(mol)
        idx_to_remove = []
        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            if not atom_idx in match:
                idx_to_remove.append(atom_idx)
        for idx in reversed(idx_to_remove):
            edit_mol.RemoveAtom(idx)
        return edit_mol


    @staticmethod
    def get_full_matrix_from_tril(tril_matrix: np.ndarray, 
                                  n: int) -> np.ndarray:
        """Get the square full matrix from a lower triangular "matrix"

        :param tril_matrix: lower triangular "matrix"
        :type tril_matrix: np.ndarray
        :param n: Number of elements (same number of rows and columns)
        :type n: int
        :return: Full matrix
        :rtype: np.ndarray
        """
        matrix = np.zeros((n, n))
        i=1
        j=0
        for v in tril_matrix:
            matrix[i, j] = matrix[j, i] = v
            j = j + 1
            if j == i:
                i = i + 1
                j = 0
        return matrix


    def compute_values(self,
                       input_list: np.ndarray
                       ) -> Sequence[float]:
        return input_list
 