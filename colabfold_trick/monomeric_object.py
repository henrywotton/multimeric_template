#
# 
# 
# 
# Author: Dingquan Yu
# 
# 
# 
# 

"""Function for parsing feature dicts"""

from genericpath import isfile
import pickle as pkl
import os 
import sys

class MonomericObject:
    """Class that store monomeric features
    
    Arguments:
    feature_dir: the directory to the corresponding feature dictionaries
    """
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        self.feature_dict = dict()
    
    def check_dir_exist(self):
        """check if the directory exists"""
        try:
            os.path.isdir(self.feature_dir)
        except:
            raise FileExistsError(f"{self.feature_dir} does not exist. Please check your input.")

        else:
            return True

    def parse_features(self):
        """method to parse all features"""
        pkl_path = os.path.join(self.feature_dir,'msas/feature_dict.pkl')
        try:
            os.path.isfile(pkl_path)
        except:
            raise FileNotFoundError(f"{pkl_path} does not exist")
        else:
            self.feature_dict = pkl.load(open(pkl_path,'rb'))
    