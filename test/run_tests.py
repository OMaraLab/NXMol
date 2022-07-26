import unittest
from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D
from chemistry_data_structure.tools import gen_tautomer_trans_structure_2D


def get_start_and_end_structures(t1_fp, t2_fp, net_charge):
    """
    Parse pdb files for test tautomers.
    :param t1_fp:
    :param t2_fp:
    :param net_charge:
    :return:
    """

    with open(t1_fp, "r") as t1_file:
        t1 = pdb_to_Molecule3D(t1_file.read(), net_charge=net_charge, assign_bond_orders_and_charges=True)

    with open(t2_fp, "r") as t2_file:
        t2 = pdb_to_Molecule3D(t2_file.read(), net_charge=net_charge, assign_bond_orders_and_charges=True)

    print("T1 Nodes: ", t1.graph.nodes)
    print("T1 Edges: ", t1.graph.edges)
    print("T1 Bond Orders: ", t1.bond_orders)
    print("T1 Formal Charges: ", t1.formal_charges)
    print("T1 Non Bonded Electrons: ", t1.non_bonded_electrons)
    print("T2 Nodes: ", t2.graph.nodes)
    print("T2 Edges: ", t2.graph.edges)
    print("T2 Bond Orders: ", t2.bond_orders)
    print("T2 Formal Charges: ", t2.formal_charges)
    print("T2 Non Bonded Electrons: ", t2.non_bonded_electrons)

    return t1, t2


class TransitionStructureTest(unittest.TestCase):

    def test_nicorandil_tautomers(self):

        t1, t2 = get_start_and_end_structures("data/pdb/nicorandil_t1.pdb", "data/pdb/nicorandil_t2.pdb", 0)

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)

        self.assertEqual(True, False)

    def test_formamide_tautomers(self):

        # open pdb files for test tautomers
        t1, t2 = get_start_and_end_structures("data/pdb/formamide_t1.pdb", "data/pdb/formamide_t2.pdb", 0)

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)



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
