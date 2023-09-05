import unittest

import networkx as nx
import pickle as pk
# import sys
# sys.path.append('../src/chemistry_data_structure/')
import numpy as np
from chemistry_data_structure.helpers.vector_calculations import place_h_using_ilp
from chemistry_data_structure.objects.atom_bond import Atom3D, Bond3D
from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D, GAMESS_to_Molecule3D
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
        self.assertEqual(list(trans_mol.bonds),
                         [('C1', 'N1'), ('C1', 'O1'), ('C1', 'H1'), ('O1', 'H2'), ('N1', 'H3'), ('N1', 'H4')])
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


class JmolStereoTests(unittest.TestCase):

    def test_stereo_assignments(self):

        test_mols = ['alanine', 'esketamine']
        test_chiral_centers = [{'C1': 'S'}, {'C3': 'S'}]
        i = 0
        for mol_name in test_mols:
            fname = f"../test/data/pdb/{mol_name}.pdb"
            with open(fname, 'r') as pdb_file:
                pdb_str = pdb_file.read()
            mol = pdb_to_Molecule3D(pdb_str, mol_name=mol_name, net_charge=0, assign_bond_orders_and_charges=True)
            self.assertDictEqual(test_chiral_centers[i], getChiralCenters(mol))
            i += 1


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

        print("Rings: ", mol.rings)

        self.assertEqual(mol.rings, [('C8', 'N3', 'C7', 'C6', 'C5', 'C4')])

    def test_aromaticity(self):
        with open("data/pdb/nicorandil_t1.pdb", "r") as t1_file:
            mol = pdb_to_Molecule3D(t1_file.read(), net_charge=0, assign_bond_orders_and_charges=True)

        print("Aromatic Atoms: ", mol.aromatic_atoms)
        print("Aromatic Bonds: ", mol.aromatic_bonds)

        self.assertEqual(set(mol.aromatic_atoms), {'C8', 'N3', 'C7', 'C6', 'C5', 'C4'})

    def test_ilp_h_placement_multi(self):

        # create a dummy methyl molecule and keep adding hydrogens to it
        mol = Molecule3D()
        mol.add_atom(Atom3D('P1', 'P', (1.0, 1.0, 1.0)))
        # mol.add_atom(Atom3D('P1', 'P', (0.0, 0.0, 0.0)))

        points = [np.array(mol.get_atom('P1').coordinates)]

        # with iterations of adding hydrogens, output the pdbStr of the molecule
        for i in range(5):

            print(f"\nPlacing Hydrogen H{i}")

            h_name = f'H{i}'
            coords = place_h_using_ilp(points, debug=True)
            points.append(np.array(coords))
            mol.add_atom(Atom3D(h_name, 'H', coords))
            mol.add_bond('P1', h_name, Bond3D({'P1', h_name}))

            # output pdb
            with open(f"tests/out/pdb/h_placement_test_{i+1}H.pdb", "w") as out_file:
                out_file.write(mol.pdbStr())

    def test_ilp_h_placement_single(self):

        # create a dummy methyl molecule and keep adding hydrogens to it
        mol = Molecule3D()
        mol.add_atom(Atom3D('P1', 'P', (0.0, 0.0, 0.0)))
        mol.add_atom(Atom3D('C1', 'C', (1.0, 0.0, 0.0)))
        mol.add_bond('P1', 'C1', Bond3D({'C1', 'P1'}))

        points = [np.array(mol.get_atom('P1').coordinates),
                  np.array(mol.get_atom('C1').coordinates)]

        # with iterations of adding hydrogens, output the pdbStr of the molecule
        coords = place_h_using_ilp(points, debug=True)
        points.append(np.array(coords))
        mol.add_atom(Atom3D('H1', 'H', coords))
        mol.add_bond('P1', 'H1', Bond3D({'P1', 'H1'}))

        # output pdb
        with open(f"tests/out/pdb/h_placement_test_single_H.pdb", "w") as out_file:
            out_file.write(mol.pdbStr())

        self.assertEqual(mol.get_atom('H1').coordinates, (-1.0, 0.0, 0.0))

    def test_marked_atom_drawing(self):

        mol_name = 'tadalafil'

        with open(f"tests/data/pdb/{mol_name}.pdb", "r") as t1_file:
            mol = pdb_to_Molecule3D(t1_file.read(), net_charge=0, assign_bond_orders_and_charges=True)

        print('Aromatic Atoms: ', mol.aromatic_atoms)

        mol.get_fragments_around_elements({'O', 'N'}, 2, restrict_arom=True)

        # TODO: fix formatting, add aromaticity restriction, remove hydrogens...
        # TODO: fix scaling of fonts etc. with image size

        font_sizes = {'node': 6, 'edge': 8, 'label': None}
        offsets = (0.8, 0.4)

        # display plots
        mol.draw_graph(font_sizes=font_sizes,
                       save_fp=f"tests/imgs/{mol_name}.png",
                       show=False,
                       node_label_mode='element',
                       node_size=140,  # TODO: we need to make this relative to the size of the graph
                       draw_marked_atoms=True)

        # TODO: fix offsets of node labels, and node sizes etc.

    def test_chirality_drawing(self):

        return


# TODO: there is no field_fitting module now? Callum ples fix or remove...
# class FieldFitTests(unittest.TestCase):
#
#     from chemistry_data_structure.helpers.field_fitting import partial_charge_fit
#
#     molecules = [GAMESS_to_Molecule3D(open(path).read()) for path in
#                  [f'data/qm/{molid}/wB97X_631Gd_SMD_water.out' for molid in [939674, 939678, 939684, 1162430]]]
#
#     ## load the reference case
#     ff_output_paths = [f'data/qm/{molid}/alkane_raw_X_[wB97X_631Gd_SMD_water].pkl' for molid in
#                        [939674, 939678, 939684, 1162430]]
#     reference_data = [pk.load(open(path, 'rb')) for path in ff_output_paths]
#
#     verbose = True
#
#     ### validating the single molecule fitting is operating correctly
#     for m, ref in zip(molecules, reference_data):
#         partial_charge_fit([m], {m: 0}, verbose=verbose)
#         if verbose:
#             print(f'\n\n\nff: {list(ref["fits"].values())[0]["rmsd"][0]}\npy: {m.partialChargeRMSD()}')
#             print('Atom Name\tff charge\tpython charge\tresid')
#         for ff_atom_name, ref_atom_data in list(ref['fits'].values())[0]['sites'].items():
#             ref_charge = ref_atom_data['fit_result']['charge'][0]
#             element, ff_id = ff_atom_name.split('_')
#             atb_name = f'{element}{int(ff_id) + 1}'
#             if verbose:
#                 print(f'{atb_name}' + '\t\t{0:.6f}\t{1:.6f}\t{2:.10f}'.format(
#                     m.get_atom(atb_name).partial_charge,
#                     float(ref_charge),
#                     abs(m.get_atom(atb_name).partial_charge - float(ref_charge))
#                 )
#                       )
#
#
#     for m, ref in zip(molecules, reference_data):
#         if verbose:
#             print(f'\n\n\nff: {list(ref["fits"].values())[0]["rmsd"][0]}\npy: {m.partialChargeRMSD()}')
#             print('Atom Name\tff charge\tpython charge\tresid')
#         for ff_atom_name, ref_atom_data in list(ref['fits'].values())[0]['sites'].items():
#             ref_charge = ref_atom_data['fit_result']['charge'][0]
#             element, ff_id = ff_atom_name.split('_')
#             atb_name = f'{element}{int(ff_id) + 1}'
#             if verbose:
#                 print(f'{atb_name}' + '\t\t\t{0:.6f}\t{1:.6f}\t{2:.10f}'.format(
#                     m.get_atom(atb_name).partial_charge,
#                     float(ref_charge),
#                     abs(m.get_atom(atb_name).partial_charge - float(ref_charge))
#                 )
#                       )


if __name__ == '__main__':
    unittest.main()
