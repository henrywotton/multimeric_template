#
# Author: Dingquan Yu 
# 
# #
import numpy as np
import os

from numpy import dtype 
from alphafold.data import pipeline
from Bio import SeqIO
import pickle as pkl

class MSAFeatures:
    """
    A class that creates msa features from a3m files
    
    Args:

    """

    def __init__(self,feature_dir) -> None:
        self.processed_feature_path = os.path.join(feature_dir,"processed_feature.pkl")
        self.processed_dict = pkl.load(open(self.processed_feature_path,'rb'))
        self.sequence = ''
        self.seq_lens = []
        self.concatenate_fasta = os.path.join(feature_dir,"concatenated.fasta") 
        self.msa_feature = dict()
        pass

    def prepare_sequences(self):
        """prepare orig_sequence and sequence"""

        records = list(SeqIO.parse(self.concatenate_fasta,format="fasta"))
        for i in range(len(records)):
            r = records[i]
            self.sequence = self.sequence + r.seq
            self.seq_lens.append(len(r.seq))

    def make_features(self):
        
        self.msa_feature = pipeline.make_sequence_features(self.sequence,"no_description",len(self.sequence))
        
        # update the residue index
        new_residue_index = np.array([])
        for i in range(len(self.seq_lens)):
            if i == 0:
                curr_seq_len = self.seq_lens[i]
                curr_region = np.arange(0,curr_seq_len)
                new_residue_index = np.concatenate((new_residue_index,curr_region),axis=0)
            else:
                curr_start = new_residue_index[-1] +1 # current starting point 
                curr_seq_len = self.seq_lens[i]
                curr_region = np.arange(curr_start + 200,curr_start + 200 + curr_seq_len)
                new_residue_index = np.concatenate((new_residue_index,curr_region),axis=0)
        
        update_dict = {
            "msa": self.processed_dict['msa'],
            "deletion_matrix_int" : self.processed_dict['deletion_matrix'],
            "num_alignments" : np.array([self.processed_dict['num_alignments'].tolist()]*len(self.sequence),dtype=np.int32),
            "msa_uniprot_accession_identifiers" : np.array("",dtype=np.object_),
            "msa_species_identifiers": np.array("",dtype=np.object_),
            'residue_index' : new_residue_index.astype(np.int32)
        }
        self.msa_feature.update(update_dict)
        