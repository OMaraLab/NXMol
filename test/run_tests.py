import unittest
from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D
from chemistry_data_structure.tools import gen_tautomer_trans_structure_2D

class TransitionStructureTest(unittest.TestCase):

    def test_nicorandil_tautomers(self):

        # open pdb files for test tautomers
        with open("data/pdb/nicorandil_t1.pdb", "r") as t1_file:
            t1 = pdb_to_Molecule3D(t1_file.read())
            print(t1.graph.nodes)
            print(t1.graph.edges)

        with open("data/pdb/nicorandil_t2.pdb", "r") as t2_file:
            t2 = pdb_to_Molecule3D(t2_file.read())
            print(t2.graph.nodes)
            print(t2.graph.edges)

        #t1.draw_graph()
        #t2.draw_graph()

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)

    def test_formamide_tautomers(self):
        # open pdb files for test tautomers
        with open("data/pdb/formamide_t1.pdb", "r") as t1_file:
            t1 = pdb_to_Molecule3D(t1_file.read())

        with open("data/pdb/formamide_t2.pdb", "r") as t2_file:
            t2 = pdb_to_Molecule3D(t2_file.read())

        print("T1 Nodes: ", t1.graph.nodes)
        print("T1 Edges: ", t1.graph.edges)
        print("T2 Nodes: ", t2.graph.nodes)
        print("T2 Edges: ", t2.graph.edges)

        #t1.draw_graph()
        #t2.draw_graph()

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)
        #trans_mol = gen_config_trans_struct_2D(t1, t1)



        self.assertEqual(True, False)


class ParserTest(unittest.TestCase):

    def test_pdb_parsing(self):
        self.assertEqual(True, False)


class NetworkxTests(unittest.TestCase):

    def test_isomorphism(self):
        import networkx as nx
        from networkx.algorithms import isomorphism

        G1 = nx.path_graph(4)
        G2 = nx.path_graph(4)

        GM = isomorphism.GraphMatcher(G1, G2)
        self.assertEqual(True, GM.is_isomorphic())


if __name__ == '__main__':
    unittest.main()
