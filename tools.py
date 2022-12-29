"""
Contains methods for performing transformations on molecular entities etc.
"""
import networkx as nx
import copy

from chemistry_data_structure.helpers.chem import VALENCE_ELECTRONS
from chemistry_data_structure.objects.atom_bond import Atom2D, Bond2D
from chemistry_data_structure.objects.base_objects import _2DChemicalObj
from chemistry_data_structure.objects.molecular_entity import Molecule2D
from networkx.algorithms import isomorphism

from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D


def get_start_and_end_structures(t1_pdb_fp, t2_pdb_fp, net_charge):
    """
    Parse pdb files to get start and end structures for tautomer transition.
    :param t1_pdb_fp:
    :param t2_pdb_fp:
    :param net_charge:
    :return:
    """

    with open(t1_pdb_fp, "r") as t1_file:
        t1 = pdb_to_Molecule3D(t1_file.read(), net_charge=net_charge, assign_bond_orders_and_charges=True)

    with open(t2_pdb_fp, "r") as t2_file:
        t2 = pdb_to_Molecule3D(t2_file.read(), net_charge=net_charge, assign_bond_orders_and_charges=True)

    # print("T1 Nodes: ", t1.graph.nodes)
    # print("T1 Edges: ", t1.graph.edges)
    # print("T1 Bond Orders: ", t1.bond_orders)
    # print("T1 Formal Charges: ", t1.formal_charges)
    # print("T1 Non Bonded Electrons: ", t1.non_bonded_electrons)
    # print("T2 Nodes: ", t2.graph.nodes)
    # print("T2 Edges: ", t2.graph.edges)
    # print("T2 Bond Orders: ", t2.bond_orders)
    # print("T2 Formal Charges: ", t2.formal_charges)
    # print("T2 Non Bonded Electrons: ", t2.non_bonded_electrons)

    return t1, t2


def gen_tautomer_trans_structure_2D(mol_start: Molecule2D,
                                    mol_end: Molecule2D) -> Molecule2D:
    """
    Generates a transition molecule representation, from one molecular graph
    to another. This is used to represent configurational isomerism transitions in 2D,
    of tautomers.
    :param mol_start: the starting Molecule2D
    :param mol_end: the ending Molecule2D
    :return: the transition structure, as a Molecule2D
    """

    # assertions
    # all atoms have formal charge attributes
    # all bonds have bond order attributes
    # an isomorphism exists between graphs
    # all heavy atoms are the same

    def atoms_equal(a1, a2):
        return a1.element == a2.element

    def bonds_equal(b1, b2):
        return b1.order == b2.order

    # 1. get a mapping between start and end backbone structure
    G1 = mol_start.get_backbone_graph()
    G2 = mol_end.get_backbone_graph()
    GM = isomorphism.GraphMatcher(G1, G2, node_match=atoms_equal, edge_match=bonds_equal)
    assert GM.is_isomorphic(), "Backbone structures of two molecules not the same."
    backbone_mapping = GM.mapping
    print("Backbone Mapping: ", backbone_mapping)
    reverse_mapping = dict((v, k) for k, v in backbone_mapping.items())

    # print("Mapping: ", backbone_mapping)
    # print("Reverse Mapping: ", reverse_mapping)

    # get bond order differences between edges and formal charge differences
    # in heavy atom bonded structure
    # get difference in number of attached hydrogens of each heavy atom
    t1_h_counts = mol_start.get_neighbour_element_counts('H')
    t2_h_counts = {reverse_mapping[atom_id]: count
                   for atom_id, count in mol_end.get_neighbour_element_counts('H').items()
                   if mol_end.get_atom(atom_id).element != 'H'}
    delta_h_counts = {atom_id: t2_h_counts[atom_id] - t1_h_counts[atom_id] for atom_id in t2_h_counts}

    # get heavy atoms and their bonds
    t1_heavy_atoms = mol_start.get_heavy_atoms()
    t1_heavy_atom_bonds = mol_start.get_heavy_atom_bonds()
    t2_heavy_atoms = mol_end.get_heavy_atoms()
    t2_heavy_atom_bonds = mol_end.get_heavy_atom_bonds()

    # formal charges
    t1_formal_charges = mol_start.get_formal_charges(t1_heavy_atoms)
    t2_formal_charges = {reverse_mapping[atom_id]: formal_charge
                         for atom_id, formal_charge in mol_end.get_formal_charges(t2_heavy_atoms).items()}
    delta_formal_charges = {atom_id: t2_formal_charges[atom_id] - t1_formal_charges[atom_id]
                            for atom_id in t1_formal_charges.keys()}

    # non-bonded electrons
    t1_nbes = mol_start.get_non_bonded_electrons(t1_heavy_atoms)
    t2_nbes = {reverse_mapping[atom_id]: nbes
               for atom_id, nbes in mol_end.get_non_bonded_electrons(t2_heavy_atoms).items()}
    delta_nbes = {atom_id: t2_nbes[atom_id] - t1_nbes[atom_id]
                  for atom_id in t1_nbes.keys()}

    # valences
    t1_valences = mol_start.get_valences(t1_heavy_atoms)
    t2_valences = {reverse_mapping[atom_id]: valence
                   for atom_id, valence in mol_end.get_valences(t2_heavy_atoms).items()}
    delta_valences = {atom_id: t2_valences[atom_id] - t1_valences[atom_id]
                      for atom_id in t1_valences.keys()}

    # hybridisations
    t1_hybridisations = mol_start.get_hybridisations(t1_heavy_atoms)
    t2_hybridisations = {reverse_mapping[atom_id]: hybridisation
                         for atom_id, hybridisation in mol_end.get_hybridisations(t2_heavy_atoms).items()}
    delta_hybridisations = {atom_id: t2_hybridisations[atom_id] - t1_hybridisations[atom_id]
                            for atom_id in t1_hybridisations.keys()}

    # conjugations
    t1_conjugations = mol_start.get_conjugations(t1_heavy_atoms)
    t2_conjugations = {reverse_mapping[atom_id]: conjugation
                         for atom_id, conjugation in mol_end.get_conjugations(t2_heavy_atoms).items()}
    delta_conjugations = {atom_id: t2_conjugations[atom_id] - t1_conjugations[atom_id]
                            for atom_id in t1_conjugations.keys()}

    # bond orders
    t1_bond_orders = mol_start.get_bond_orders(t1_heavy_atom_bonds)
    t2_bond_orders = {frozenset((reverse_mapping[list(bond_id)[0]], reverse_mapping[list(bond_id)[1]])): order
                      for bond_id, order in mol_end.get_bond_orders(t2_heavy_atom_bonds).items()}
    delta_bond_orders = {bond_id: t2_bond_orders[bond_id] - t1_bond_orders[bond_id]
                         for bond_id in t1_bond_orders.keys()}

    # TODO: look... there's definitely a more efficient way to do all this, but its neat and im lazy^^^

    # print("t1_formal_charges: ", t1_formal_charges)
    # print("t2_formal_charges: ", t2_formal_charges)
    # print("t2 unmapped charges: ", mol_end.get_formal_charges(t2_heavy_atoms))
    # print("Delta formal charges: ", delta_formal_charges)
    assert t1_formal_charges.keys() == t2_formal_charges.keys()
    assert t1_nbes.keys() == t2_nbes.keys()

    # print("t1_bond_orders: ", t1_bond_orders)
    # print("t2_bond_orders: ", t2_bond_orders)
    # print("t2 unmapped bond orders: ", mol_end.get_bond_orders(t2_heavy_atom_bonds))
    # print("Delta bond_orders: ", delta_bond_orders)
    assert t1_bond_orders.keys() == t2_bond_orders.keys()

    # print("t1_h_counts: ", t1_h_counts)
    # print("t2_h_counts: ", t2_h_counts)
    # print("Delta h counts: ", delta_h_counts)

    # create new graph with atom/edge labels from G1, with attributes as the deltas determined above
    # based on number of hydrogens difference at each heavy atom, add newly indexed hydrogens to
    # heavy atoms, with edge labels as -1 if removed, 0 if the same and +1 if added

    # get heavy atoms for trans mol
    n_id = 0
    trans_atoms = []
    for atom_id in t1_heavy_atoms:
        trans_atoms.append(
            Atom2D(
                index={'name': atom_id, 'nid': n_id},
                name=atom_id,
                element=mol_start.get_atom(atom_id).element,
                valence_electrons=VALENCE_ELECTRONS[mol_start.get_atom(atom_id).element.upper()],
                formal_charge=delta_formal_charges[atom_id],
                non_bonded_electrons=delta_nbes[atom_id],
                valence=delta_valences[atom_id],
                hybridisation=delta_hybridisations[atom_id],
                is_conjugated=delta_conjugations[atom_id],
            )
        )
        n_id += 1

    # add heavy atom bonds with delta bond orders
    trans_bonds = []
    for bond_id in t1_bond_orders.keys():
        trans_bond = Bond2D(order=delta_bond_orders[bond_id])
        a1_ind, a2_ind = list(bond_id)
        trans_bonds.append((a1_ind, a2_ind, trans_bond))

    # add bonds to hydrogens with delta bond orders
    h_id = 1
    for atom_id in t1_heavy_atoms:

        # add all static and transition hydrogens
        num_trans_h = max([t1_h_counts[atom_id], t2_h_counts[atom_id]])
        for i in range(0, num_trans_h):

            h_name = f"H{h_id}"

            # add hydrogen
            trans_atoms.append(
                Atom2D(
                    index={'name': h_name, 'nid': n_id},
                    name=h_name,
                    element='H',
                    formal_charge=0,
                    non_bonded_electrons=0,
                    valence=0,
                    hybridisation=0,
                    is_conjugated=0,
                    is_aromatic=0,
                )
            )
            n_id += 1

            if i < num_trans_h - abs(delta_h_counts[atom_id]):
                # add static hydrogen bonds
                trans_h_bond = Bond2D(order=0)
            else:
                # add transition hydrogen bond
                if delta_h_counts[atom_id] < 0:
                    # lost hydrogen, assign negative bond order
                    trans_h_bond = Bond2D(order=-1)
                else:
                    # added hydrogen, assign positive bond order
                    trans_h_bond = Bond2D(order=1)

            trans_bonds.append((atom_id, h_name, trans_h_bond))
            h_id += 1

    trans_mol = Molecule2D(
        trans_atoms,
        trans_bonds,
        name="transition",
    )

    return trans_mol


if __name__ == "__main__":
    # DEBUG ONLY
    mol_1 = Molecule2D()
