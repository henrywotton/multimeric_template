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
        self.new_aatype = np.array([0])
        self.num_template=0
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
        
        for m in self.monomers:
            self.num_template += m.template_feature_dict['template_aatype'].shape[0]
        

    def create_concatenated_template_features(self) -> dict:
        """It is only for 2 subunits for now so it is hard coded"""
        seq_length_1 = self.monomers[0].template_feature_dict['template_aatype'].shape[1]
        seq_length_2 = self.monomers[1].template_feature_dict['template_aatype'].shape[1]
        num_template_1 = self.monomers[0].template_feature_dict['template_aatype'].shape[0]
        num_template_2 = self.monomers[1].template_feature_dict['template_aatype'].shape[0]
        
        concat_aatype_1,concat_atom_mask_1,concat_atom_pos_1 = self.concatenate_features(idx=0,
        seq_length=seq_length_2,num_template=num_template_1)
        
        concat_aatype_2,concat_atom_mask_2,concat_atom_pos_2 = self.concatenate_features(idx=1,
        seq_length=seq_length_1,num_template=num_template_2)

        def prepare_output(array1,array2):
            return np.concatenate((array1,array2),axis=0)

        output_aatype = prepare_output(concat_aatype_1,concat_aatype_2)
        output_atom_mask = prepare_output(concat_atom_mask_1,concat_atom_mask_2)
        output_atom_pos = prepare_output(concat_atom_pos_1,concat_atom_pos_2)
        output_domain_names = self.concatenate_domain_names()

        return {
            "template_all_atom_positions" : output_atom_pos,
            "template_all_atom_masks" : output_atom_mask,
            "template_sequence" : [f"none".encode()] * self.num_template,
            "template_aatype" : output_aatype,
            "template_domain_names" : output_domain_names,
            "template_release_date": [f"none".encode()] * self.num_template,
            "template_sum_probs": np.zeros([self.num_template], dtype=np.float32),
            "template_sequence" : [f"none".encode()] * self.num_template
        }
    
    def create_mock_aatype(self,seq_length,num_template):
        """create mock sequence then turn it into one-hot. Tile the one-hot to be the same dimension as original """
        mock_template_sequence = "-" * seq_length
        mock_template_aatype = templates.residue_constants.sequence_to_onehot(
        mock_template_sequence, templates.residue_constants.HHBLITS_AA_TO_ID)
        tiled_template_aatype = np.tile(
            mock_template_aatype,(num_template,1,1)
        )

        return tiled_template_aatype

    def create_mock_atom_mask(self,seq_length,num_template):
        """create mock atom mask it"""
        mock_atom_mask = np.zeros((seq_length, templates.residue_constants.atom_type_num))
        tiled_atom_mask = np.tile(mock_atom_mask,(num_template,1,1))
        return tiled_atom_mask

    def create_mock_atom_pos(self,seq_length,num_template):
        """create mock atom positions """
        mock_atom_pos = np.zeros((seq_length, templates.residue_constants.atom_type_num,3)) 
        tiled_atom_pos = np.tile(mock_atom_pos,(num_template,1,1,1))
        return tiled_atom_pos

    def concatenate_features(self,idx,seq_length,num_template):
        """
        a more generic function to concatenate features
        
        Args:
        feature_name: name of the feature that to be concatenated
        idx: idx of the monomeric object in the list
        """
        orig_aatype = self.monomers[idx].template_feature_dict['template_aatype']
        orig_atom_mask = self.monomers[idx].template_feature_dict['template_all_atom_masks']
        orig_atom_pos = self.monomers[idx].template_feature_dict['template_all_atom_positions']
        
        
        tiled_mock_aatype = self.create_mock_aatype(seq_length=seq_length,
        num_template=num_template)
        tiled_atom_mask = self.create_mock_atom_mask(seq_length=seq_length,
        num_template=num_template)
        tiled_atom_pos = self.create_mock_atom_pos(seq_length=seq_length,
        num_template=num_template)
        
        if (idx %2 )==0:
            concat_aatype = np.concatenate((orig_aatype,tiled_mock_aatype),axis=1)
            concat_atom_mask = np.concatenate((orig_atom_mask,tiled_atom_mask),axis=1)
            concat_atom_pos = np.concatenate((orig_atom_pos,tiled_atom_pos),axis=1)
        else:
            concat_aatype = np.concatenate((tiled_mock_aatype,orig_aatype),axis=1)
            concat_atom_mask = np.concatenate((tiled_atom_mask,orig_atom_mask),axis=1)
            concat_atom_pos = np.concatenate((tiled_atom_pos,orig_atom_pos),axis=1)

        return concat_aatype,concat_atom_mask,concat_atom_pos


    def concatenate_domain_names(self):
        """concatenate domain names"""

        conct_domains = []
        for m in self.monomers:
            conct_domains = np.concatenate((conct_domains,m.template_feature_dict['template_domain_names']))

        return conct_domains
