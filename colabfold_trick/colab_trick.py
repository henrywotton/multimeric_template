#
# Author : Dingquan Yu
# #
from prepare_input import MultimericInput
import os 
from absl import app
from absl import flags
from absl import logging
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union, Optional

flags.DEFINE_string("precomputed_dir",None,help="directory to precomputed features")
flags.DEFINE_string("processed_dir",None,help="directory to processed features")
flags.DEFINE_string("protein_names",None,help="name of the pair: A-B")
flags.DEFINE_string("output_dir",None,help="output directory")
flags.DEFINE_boolean("use_template",True,help="use templates or not")

FLAGS = flags.FLAGS

def create_input_dict(base_precomputed_dir,base_processed_dir,input_protein_names):
   
    protein_names = []
    protein_names.append(input_protein_names.split("-")[0])
    protein_names.append(input_protein_names.split("-")[1])

    multimetic_input = MultimericInput(base_precomputed_dir,protein_names)
    multimetic_input.create_monomeric_objects()
    output_msa_feature_dict = multimetic_input.create_msa_dict(base_processed_dir)

    if not FLAGS.use_template:
        output_template_dict = multimetic_input.create_mock_template_features()
    else:
        output_template_dict = multimetic_input.create_concatenated_template_features()

    return {**output_msa_feature_dict,**output_template_dict}


def main(argv):
    """main function """

    feature_dict = create_input_dict(FLAGS.precomputed_dir,FLAGS.processed_dir,FLAGS.protein_names)
    dump_path = os.path.join(FLAGS.output_dir,'colabfold_feature.pkl')
    pickle.dump(feature_dict,open(dump_path,'wb'))

if __name__ == '__main__':
    app.run(main)
