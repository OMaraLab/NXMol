import unittest

import networkx as nx
import pickle as pk

from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D, GAMESS_to_Molecule3D
from chemistry_data_structure.tools import gen_tautomer_trans_structure_2D, get_start_and_end_structures
from chemistry_data_structure.helpers.field_fitting import MoleculeFieldFitter


class TransitionStructureTest(unittest.TestCase):

    def test_nicorandil_tautomers(self):
        t1, t2 = get_start_and_end_structures("data/pdb/nicorandil_t1.pdb", "data/pdb/nicorandil_t2.pdb", 0)

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)

        # display plots
        t1.draw_graph()
        t2.draw_graph()
        trans_mol.draw_graph()

        self.assertEqual(True, True)

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
        self.assertEqual(list(trans_mol.bonds),
                         [('C1', 'N1'), ('C1', 'O1'), ('C1', 'H1'), ('O1', 'H2'), ('N1', 'H3'), ('N1', 'H4')])
        self.assertEqual(trans_mol.bond_orders, {frozenset({'N1', 'C1'}): 1, frozenset({'O1', 'C1'}): -1,
                                                 frozenset({'H1', 'C1'}): 0, frozenset({'O1', 'H2'}): 1,
                                                 frozenset({'N1', 'H3'}): 0, frozenset({'N1', 'H4'}): -1})

        # display plots
        t1.draw_graph()
        t2.draw_graph()
        trans_mol.draw_graph()

    def test_glycine_tautomers(self):
        t1, t2 = get_start_and_end_structures("data/pdb/glycine_t1.pdb", "data/pdb/glycine_t2.pdb", 0)

        # gen transition structure
        trans_mol = gen_tautomer_trans_structure_2D(t1, t2)

        # make fixed heavy atom node positions
        heavy_atom_keys = t2.get_backbone_graph().nodes
        fixed_heavy_atom_positions = {'N1': (0, 1),
                                      'C1': (2, 1),
                                      'C2': (4, 1),
                                      'O1': (6, 2),
                                      'O2': (6, 0)}

        t2_fixed_heavy_atom_positions = {'N1': (0, 1),
                                         'C2': (2, 1),
                                         'C1': (4, 1),
                                         'O1': (6, 2),
                                         'O2': (6, 0)}

        print(heavy_atom_keys)

        # display plots
        t1.draw_graph(fixed_heavy_atom_positions)
        t2.draw_graph(t2_fixed_heavy_atom_positions)
        trans_mol.draw_graph(fixed_heavy_atom_positions)

        t1.draw_graph(fixed_heavy_atom_positions, backbone_only=True)
        t2.draw_graph(t2_fixed_heavy_atom_positions, backbone_only=True)

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


class MultiFitterTests(unittest.TestCase):
    molecules = [GAMESS_to_Molecule3D(open(path).read()) for path in
                 [f'data/qm/{molid}/wB97X_631Gd_SMD_water.out' for molid in [939674, 939678, 939684, 1162430]]]

    ## load the reference case
    ff_output_paths = [f'data/qm/{molid}/alkane_raw_X_[wB97X_631Gd_SMD_water].pkl' for molid in
                       [939674, 939678, 939684, 1162430]]
    reference_data = [pk.load(open(path, 'rb')) for path in ff_output_paths]

    verbose = True

    ### validating the single molecule fitting is operating correctly
    for m, ref in zip(molecules, reference_data):
        m.setfitPartialCharges()
        if verbose:
            print(f'\n\n\nff: {list(ref["fits"].values())[0]["rmsd"][0]}\npy: {self.partialChargeRMSD()}')
            print('Atom Name\tff charge\tpython charge\tresid')
        for ff_atom_name, ref_atom_data in list(ref['fits'].values())[0]['sites'].items():
            ref_charge = ref_atom_data['fit_result']['charge'][0]
            element, ff_id = ff_atom_name.split('_')
            atb_name = f'{element}{int(ff_id) + 1}'
            if verbose:
                print(f'{atb_name}' + '\t\t{0:.6f}\t{1:.6f}\t{2:.10f}'.format(
                    m.get_atom(atb_name).partial_charge,
                    float(ref_charge),
                    abs(m.get_atom(atb_name).partial_charge - float(ref_charge))
                )
                      )

    fitter = MoleculeFieldFitter(molecules)
    # attempting the fit without constraints (but with the total charge since all the reference data ahs that
    fitter.generate_matrices()
    fitter.load_constraints(
        symmetry_constraints={},
        sum_constraints=[{'atoms':
                              {molecule: list(range(molecule.num_atoms))},
                          'value': 0.0}
                         for molecule in molecules]
    )
    fitter.fit()
    fitter.transfer_partial_charges()

    for m, ref in zip(molecules, reference_data):
        if verbose:
            print(f'\n\n\nff: {list(ref["fits"].values())[0]["rmsd"][0]}\npy: {m.partialChargeRMSD()}')
            print('Atom Name\tff charge\tpython charge\tresid')
        for ff_atom_name, ref_atom_data in list(ref['fits'].values())[0]['sites'].items():
            ref_charge = ref_atom_data['fit_result']['charge'][0]
            element, ff_id = ff_atom_name.split('_')
            atb_name = f'{element}{int(ff_id) + 1}'
            if verbose:
                print(f'{atb_name}' + '\t\t\t{0:.6f}\t{1:.6f}\t{2:.10f}'.format(
                    m.get_atom(atb_name).partial_charge,
                    float(ref_charge),
                    abs(m.get_atom(atb_name).partial_charge - float(ref_charge))
                )
                      )

    ## get the atoms into the correct format

    for mol_iter in range(4):
        pass


if __name__ == '__main__':
    unittest.main()
