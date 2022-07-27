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

        print("\nTransition Molecule:")
        print("Atoms: ", trans_mol.atoms)
        print("Formal Charges: ", trans_mol.formal_charges)
        print("Bonds: ", trans_mol.bonds)
        print("Bond Orders: ", trans_mol.bond_orders)

        self.assertEqual(list(trans_mol.atoms), ['C1', 'O1', 'N1', 'H1', 'H2', 'H3', 'H4'])
        self.assertEqual(trans_mol.formal_charges, {'C1': 0, 'O1': 0, 'N1': 0, 'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0})
        self.assertEqual(list(trans_mol.bonds), [('C1', 'N1'), ('C1', 'O1'), ('C1', 'H1'), ('O1', 'H2'), ('N1', 'H3'), ('N1', 'H4')])
        self.assertEqual(trans_mol.bond_orders, {frozenset({'N1', 'C1'}): 1, frozenset({'O1', 'C1'}): -1,
                                                 frozenset({'H1', 'C1'}): 0, frozenset({'O1', 'H2'}): 1,
                                                 frozenset({'N1', 'H3'}): 0, frozenset({'N1', 'H4'}): -1})


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
