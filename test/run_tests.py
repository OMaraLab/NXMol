import unittest

import networkx as nx

from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D
from chemistry_data_structure.tools import gen_tautomer_trans_structure_2D, get_start_and_end_structures


class TransitionStructureTest(unittest.TestCase):

    def test_nicorandil_tautomers(self):

        t1, t2 = get_start_and_end_structures("data/pdb/nicorandil_t1.pdb", "data/pdb/nicorandil_t2.pdb", 0)

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)

        print("\nTransition Molecule:")
        print("Atoms: ", trans_mol.atoms)
        print("Valences: ", trans_mol.valences)
        print("Non-bonded Electrons: ", trans_mol.non_bonded_electrons)
        print("Hybridisations: ", trans_mol.hybridisations)
        print("Conjugations", trans_mol.atom_conjugations)
        print("Formal Charges: ", trans_mol.formal_charges)
        print("Bonds: ", trans_mol.bonds)
        print("Bond Orders: ", trans_mol.bond_orders)

        # display plots
        t1.draw_graph()
        t2.draw_graph()
        trans_mol.draw_graph()

        self.assertEqual(True, True)

    def test_formamide_tautomers(self):

        # open pdb files for test tautomers
        t1, t2 = get_start_and_end_structures("data/pdb/formamide_t1.pdb", "data/pdb/formamide_t2.pdb", 0)

        print("Adjacency matrix of t1: \n", t1.get_graph_adj_mat())

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)

        print("\nTransition Molecule:")
        print("Atoms: ", trans_mol.atoms)
        print("Valences: ", trans_mol.valences)
        print("Non-bonded Electrons: ", trans_mol.non_bonded_electrons)
        print("Hybridisations: ", trans_mol.hybridisations)
        print("Conjugations", trans_mol.atom_conjugations)
        print("Formal Charges: ", trans_mol.formal_charges)
        print("Bonds: ", trans_mol.bonds)
        print("Bond Orders: ", trans_mol.bond_orders)

        self.assertEqual(list(trans_mol.atoms), ['C1', 'O1', 'N1', 'H1', 'H2', 'H3', 'H4'])
        self.assertEqual(trans_mol.formal_charges, {'C1': 0, 'O1': 0, 'N1': 0, 'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0})
        self.assertEqual(list(trans_mol.bonds), [('C1', 'N1'), ('C1', 'O1'), ('C1', 'H1'), ('O1', 'H2'), ('N1', 'H3'), ('N1', 'H4')])
        self.assertEqual(trans_mol.bond_orders, {frozenset({'N1', 'C1'}): 1, frozenset({'O1', 'C1'}): -1,
                                                 frozenset({'H1', 'C1'}): 0, frozenset({'O1', 'H2'}): 1,
                                                 frozenset({'N1', 'H3'}): 0, frozenset({'N1', 'H4'}): -1})

        # display plots
        # t1.draw_graph()
        # t2.draw_graph()
        # trans_mol.draw_graph()

        # just for drawing nice glycine graph
        font_sizes = {'node': 14, 'edge': 20, 'label': 20}
        save_dir = "imgs"
        fixed_heavy_atom_positions = {'N1': (0, 0),
                                      'C1': (4, 0),
                                      'O1': (8, 3)}
        t1.draw_graph(fixed_heavy_atoms=fixed_heavy_atom_positions,
                      font_sizes=font_sizes, save_fp=f"{save_dir}/formamide.png", show=True,
                      draw_formal_charges=False, draw_atom_ids=False)

    def test_glycine_tautomers(self):

        t1, t2 = get_start_and_end_structures("data/pdb/glycine_t1.pdb", "data/pdb/glycine_t2.pdb", 0)

        print("Adjacency matrix of t1: \n", t1.get_graph_adj_mat())

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)

        print("\nTransition Molecule:")
        print("Atoms: ", trans_mol.atoms)
        print("Valences: ", trans_mol.valences)
        print("Non-bonded Electrons: ", trans_mol.non_bonded_electrons)
        print("Hybridisations: ", trans_mol.hybridisations)
        print("Conjugations", trans_mol.atom_conjugations)
        print("Formal Charges: ", trans_mol.formal_charges)
        print("Bonds: ", trans_mol.bonds)
        print("Bond Orders: ", trans_mol.bond_orders)

        # make fixed heavy atom node positions
        heavy_atom_keys = t2.get_backbone_graph().nodes
        fixed_heavy_atom_positions = {'N1': (0, 1),
                                      'C1': (3, 1),
                                      'C2': (6, 1),
                                      'O1': (9, 2),
                                      'O2': (9, 0)}

        t2_fixed_heavy_atom_positions = {'N1': (0, 1),
                                          'C2': (2, 1),
                                          'C1': (4, 1),
                                          'O1': (6, 2),
                                          'O2': (6, 0)}

        print(heavy_atom_keys)

        font_sizes = {'node': 14, 'edge': 20, 'label': 20}
        save_dir = "imgs"
        t1_offsets = (0.8, 0.4)
        t2_offsets = (0.54, 0.24)
        t1_heavy_offsets = (0.3, 0.2)
        t2_heavy_offsets = (0.2, 0.2)

        # display plots
        t1.draw_graph(fixed_heavy_atom_positions, font_sizes=font_sizes, save_fp=f"{save_dir}/t1.png", show=False, offsets=t1_offsets, draw_formal_charges=False)
        t2.draw_graph(t2_fixed_heavy_atom_positions, font_sizes=font_sizes, save_fp=f"{save_dir}/t2.png", show=False, offsets=t2_offsets)
        trans_mol.draw_graph(fixed_heavy_atom_positions, font_sizes=font_sizes, save_fp=f"{save_dir}/t1t2.png", show=False, offsets=t1_offsets)

        t1.draw_graph(fixed_heavy_atom_positions, backbone_only=True, font_sizes=font_sizes, save_fp=f"{save_dir}/t1_heavy.png", show=False, offsets=t1_heavy_offsets)
        t2.draw_graph(t2_fixed_heavy_atom_positions, backbone_only=True, font_sizes=font_sizes, save_fp=f"{save_dir}/t2_heavy.png", show=False, offsets=t2_heavy_offsets)

        self.assertEqual(True, True)


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

    def test_plotting(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        with open("data/pdb/formamide_t1.pdb", "r") as t1_file:
            test_mol = pdb_to_Molecule3D(t1_file.read())
        nx.draw(test_mol.graph)
        plt.show()
        plt.cla()
        nx.draw(test_mol.graph, with_labels=True)
        plt.show()
        plt.cla()


class ChemObjectTests(unittest.TestCase):

    def test_rings(self):

        with open("data/pdb/nicorandil_t1.pdb", "r") as t1_file:
            mol = pdb_to_Molecule3D(t1_file.read(), net_charge=0, assign_bond_orders_and_charges=True)

        print("Rings: ", mol.get_rings())

        self.assertEqual(mol.get_rings(), [('C8', 'N3', 'C7', 'C6', 'C5', 'C4')])


if __name__ == '__main__':
    unittest.main()
