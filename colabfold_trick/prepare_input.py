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
import sys
from .monomeric_object import MonomericObject

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
    
    def concatnate_template_features(self):
        """concatenate template features from monomeric objects"""