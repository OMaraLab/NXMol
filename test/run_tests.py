import unittest
from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D


class TransitionStructureTest(unittest.TestCase):

    def test_nicorandil_tautomers(self):

        # open pdb files for test tautomers
        with open("data/pdb/nicorandil_t1.pdb", "r") as t1_file:
            t1 = pdb_to_Molecule3D(t1_file.read(), 0)

        self.assertEqual(True, False)


class ParserTest(unittest.TestCase):

    def test_pdb_parsing(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
