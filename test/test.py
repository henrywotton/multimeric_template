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

if __name__ == '__main__':
    unittest.main()
