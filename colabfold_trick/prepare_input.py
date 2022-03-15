#
# 
# 
# 
# Author: Dingquan Yu
# 
# 
# 

"""Functions for preparing input"""
import pickle as pkl
import os 
from alphafold.data import templates
from monomeric_object import MonomericObject
from concatenate_template_object import ConcatenatedTemplate
from msa_feature_object import MSAFeatures
import numpy as np

class MultimericInput:
    """
    Class that store input feature dictionary

    Arguments:
    feature_dir: path to the each subunit's feature dicts
    protein_names: names of the subunits to model
    """

    def __init__(self, feature_dir,protein_names) -> None:
        self.feature_dir = feature_dir
        self.protein_names = protein_names
        self.monomers = list()
        self.query_sequence = ''
        self.new_aatype = np.array([0])
        self.num_template=0
        pass

    def create_monomeric_objects(self) -> None:
        """ create monomeric objects """
        for i in range(len(self.protein_names)):
            curr_feature_dir = os.path.join(self.feature_dir,self.protein_names[i])
            m = MonomericObject(curr_feature_dir)

            if m.check_dir_exist():
                m.parse_features()
            
            self.monomers.append(m)
        
        for m in self.monomers:
            self.num_template += m.feature_dict['template_aatype'].shape[0]

    def create_concatenated_template_features(self) -> dict:
        """It is only for 2 subunits for now so it is hard coded"""
        seq_length_1 = self.monomers[0].feature_dict['template_aatype'].shape[1]
        seq_length_2 = self.monomers[1].feature_dict['template_aatype'].shape[1]
        num_template_1 = self.monomers[0].feature_dict['template_aatype'].shape[0]
        num_template_2 = self.monomers[1].feature_dict['template_aatype'].shape[0]
        
        concatenate_template_obj = ConcatenatedTemplate(self.monomers)
        concat_aatype_1,concat_atom_mask_1,concat_atom_pos_1 = concatenate_template_obj.concatenate_features(idx=0,
        seq_length=seq_length_2,num_template=num_template_1)
        
        concat_aatype_2,concat_atom_mask_2,concat_atom_pos_2 = concatenate_template_obj.concatenate_features(idx=1,
        seq_length=seq_length_1,num_template=num_template_2)

        def prepare_output(array1,array2):
            return np.concatenate((array1,array2),axis=0)

        output_aatype = prepare_output(concat_aatype_1,concat_aatype_2)
        output_atom_mask = prepare_output(concat_atom_mask_1,concat_atom_mask_2)
        output_atom_pos = prepare_output(concat_atom_pos_1,concat_atom_pos_2)
        output_domain_names = concatenate_template_obj.concatenate_domain_names()

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

    def create_mock_template_features(self) -> dict:
        """this method is triggered only when the mode is set to not to use templates"""

        for m in self.monomers:
            curr_seq = m.feature_dict['sequence'][0].decode("utf-8")
            self.query_sequence = self.query_sequence + curr_seq

        query_sequence = self.query_sequence
        num_temp = 1
        ln = (
            len(query_sequence)
            if isinstance(query_sequence, str)
            else sum(len(s) for s in query_sequence)
        )
        output_templates_sequence = "A" * ln
        output_confidence_scores = np.full(ln, 1.0)


        templates_all_atom_positions = np.zeros(
            (ln, templates.residue_constants.atom_type_num, 3)
        )
        templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
        templates_aatype = templates.residue_constants.sequence_to_onehot(
            output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
        )
        template_features = {
            "template_all_atom_positions": np.tile(
                templates_all_atom_positions[None], [num_temp, 1, 1, 1]
            ),
            "template_all_atom_masks": np.tile(
                templates_all_atom_masks[None], [num_temp, 1, 1]
            ),
            "template_sequence": [f"none".encode()] * num_temp,
            "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
            "template_confidence_scores": np.tile(
                output_confidence_scores[None], [num_temp, 1]
            ),
            "template_domain_names": [f"none".encode()] * num_temp,
            "template_release_date": [f"none".encode()] * num_temp,
            "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
        }
        return template_features


    def create_msa_dict(self,work_dir) -> dict:
        """create an MSAFeatures object and return msa feature dict"""

        msa_feature_object = MSAFeatures(work_dir)
        msa_feature_object.prepare_sequences()
        msa_feature_object.make_features()

        return msa_feature_object.msa_feature
