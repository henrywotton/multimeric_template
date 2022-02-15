import numpy as np
from alphafold.data import templates

class ConcatenatedTemplate:
    """a class that product transformed template features """
    def __init__(self,monomers) -> None:
        self.monomers = monomers
        pass


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