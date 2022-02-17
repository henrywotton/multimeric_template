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
flags.DEFINE_string("data_dir",None,help="data dir")
flags.DEFINE_integer("num_cycle",3,help="number of recycle")
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

    
    num_ensemble = 1
    model_runners = {}
    model_names = config.MODEL_PRESETS['monomer_ptm']
    
    for model_name in model_names:
        model_config = config.model_config(model_name)  
        model_config['model'].update({'num_recycle':FLAGS.num_cycle})
        logging.info("the number of recyles is {}".format(model_config['model']['num_recycle']))
        model_config.model.num_ensemble_eval = num_ensemble
        model_params = data.get_model_haiku_params(
            model_name=model_name, data_dir=FLAGS.data_dir)
        model_runner = model.RunModel(model_config, model_params)
        model_runners[model_name] = model_runner
    logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_names))
        logging.info('Using random seed %d for the data pipeline', random_seed)


    feature_dict = create_input_dict(FLAGS.precomputed_dir,FLAGS.processed_dir,FLAGS.protein_names)
    predict(model_runners=model_runners,output_dir=FLAGS.output_dir,
    feature_dict=feature_dict,random_seed=False,benchmark=False,
    fasta_name=FLAGS.protein_names,
    amber_relaxer=False)

if __name__ == '__main__':
    app.run(main)