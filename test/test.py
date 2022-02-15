import unittest
from colabfold_trick.prepare_input import MultimericInput
test_feature_dict = '/media/geoffrey/bigdata/scratch/gyu/af2_lasv_L_apms_precomputed'
test_protein_names = ['O00571','fragment_1']

class TestSum(unittest.TestCase):

    def test_create_monomeric_objects(self):
        """test creating monomeric objects"""

        test_input = MultimericInput(test_feature_dict,test_protein_names)
        test_input.create_monomeric_objects()
        self.assertIsInstance(test_input.monomers,list)
        self.assertEqual(len(test_protein_names),len(test_input.monomers))

    def test_concatenate_aatype(self):
        """test aatype concatenation"""

        test_input = MultimericInput(test_feature_dict,test_protein_names)
        test_input.create_monomeric_objects()
        output_aatype = test_input.concatenate_template_aatype()

        total_num_templates = 0
        for m in test_input.monomers:
            total_num_templates += m['template_aatype'].shape[0]
        
        total_seq_length = 0
        for m in test_input.monomers:
            total_seq_length += m['template_aatype'].shape[1]

        print(f"Total number of templates is {total_num_templates}")
        print(f"Final sequence length is {total_seq_length}")

        self.assertEqual((total_num_templates,total_seq_length,22),output_aatype.shape)


if __name__ == '__main__':
    unittest.main()
