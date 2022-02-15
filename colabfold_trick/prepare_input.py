#
# 
# 
# 
# Author: Dingquan Yu
# 
# 
# 

"""Functions for preparing input"""

from operator import concat
import pickle as pkl
import os 
import sys
from .monomeric_object import MonomericObject
import numpy as np
from alphafold.data import templates

class MultimericInput:
    """
    Class that store input feature dictionary

    Arguments:
    feature_dir: path to the each subunit's feature dicts
    protein_names: names of the subunits to model
    """

    def __init__(self, feature_dir, protein_names) -> None:
        self.feature_dir = feature_dir
        self.protein_names = protein_names
        self.monomers = list()
        self.new_aatype = np.array()
        pass

    def create_monomeric_objects(self) -> None:
        """ create monomeric objects """
        for i in range(len(self.protein_names)):
            curr_feature_dir = os.path.join(self.feature_dir,self.protein_names[i])
            print("the curret path is "+ curr_feature_dir)
            m = MonomericObject(curr_feature_dir)

            if m.check_dir_exist():
                m.parse_template_features()
            
            self.monomers.append(m)
    
    def concatenate_template_aatype(self):
        """
        concatenate template aatypess from monomeric objects
        Here because I only work on L fragments and others so the re-ordering is hard-coded 
        """
        L_template = self.monomers[-1]['template_aatype']
        host_template = self.monomers[0]['template_aatype']
        len_L_fragment = L_template.shape[1]
        len_host = host_template.shape[1]

        """create mock aatypes"""
        mock_template_sequence_for_L = "-" * len_L_fragment
        mock_template_sequence_for_host = "-" * len_host
        mock_template_aatype_for_L = templates.residue_constants.sequence_to_onehot(
        mock_template_sequence_for_L, templates.residue_constants.HHBLITS_AA_TO_ID)
        tiled_template_aatype_for_L = np.tile(
            mock_template_aatype_for_L,(L_template.shape[0],1,1)
        )
        concat_aatype_for_L = np.concatenate((tiled_template_aatype_for_L,L_template))

        mock_template_aatype_for_host =  templates.residue_constants.sequence_to_onehot(
        mock_template_sequence_for_host, templates.residue_constants.HHBLITS_AA_TO_ID)
        tiled_template_aatype_for_host = np.tile(
            mock_template_aatype_for_host,(host_template.shape[0],1,1)
        )
        concat_aatype_for_host = np.concatenate((host_template,tiled_template_aatype_for_host))

        return np.concatenate((concat_aatype_for_host,concat_aatype_for_L),axis=0)

    def concatenate_template_all_atom_mask(self):
        