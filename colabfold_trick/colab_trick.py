#
# Author : Dingquan Yu
# #
from prepare_input import MultimericInput
from predict_structure_from_processed_features import predict
import os 
from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
from alphafold.model import data
import time
from typing import Dict, Union, Optional

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

flags.DEFINE_string("precomputed_dir",None,help="directory to precomputed features")
flags.DEFINE_string("processed_dir",None,help="directory to processed features")
flags.DEFINE_string("protein_names",None,help="name of the pair: A-B")
flags.DEFINE_string("output_dir",None,help="output directory")

FLAGS = flags.FLAGS

def create_input_dict(base_precomputed_dir,base_processed_dir,input_protein_names):
   
    protein_names = []
    protein_names.append(input_protein_names.split("-")[0])
    protein_names.append(input_protein_names.split("-")[1])

    multimetic_input = MultimericInput(base_precomputed_dir,protein_names)
    multimetic_input.create_monomeric_objects()
    work_dir = os.path.join(base_processed_dir,f"{protein_names[0]}/{protein_names[1]}")
    output_msa_feature_dict = multimetic_input.create_msa_dict(work_dir)
    output_template_dict = multimetic_input.create_concatenated_template_features()

    return {**output_msa_feature_dict,**output_template_dict}


def main(argv):
    """main function """

    feature_dict = create_input_dict(FLAGS.precomputed_dir,FLAGS.processed_dir,FLAGS.protein_names)
    dump_path = os.path.join(FLAGS.output_dir,'processed_feature.pkl')
    pickle.dump(feature_dict,open(dump_path,'wb'))

if __name__ == '__main__':
    app.run(main)