"""
Contains methods for performing transformations on molecular entities etc.
"""
import networkx as nx
import copy

from chemistry_data_structure.objects.atom_bond import Atom2D, Bond2D
from chemistry_data_structure.objects.molecular_entity import Molecule2D
from networkx.algorithms import isomorphism


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

    # 1. get a mapping between start and end backbone structure
    G1 = mol_start.get_backbone_graph()
    G2 = mol_end.get_backbone_graph()
    GM = isomorphism.GraphMatcher(G1, G2, node_match=atoms_equal)
    assert GM.is_isomorphic(), "Backbone structures of two molecules not the same."
    print("Is Isomorphic: ", GM.is_isomorphic())
    backbone_mapping = GM.mapping
    reverse_mapping = dict((v, k) for k, v in backbone_mapping.items())
    print("Mapping: ", backbone_mapping)
    print("Reverse Mapping: ", reverse_mapping)

    # get bond order differences between edges and formal charge differences
    # in heavy atom bonded structure
    # get difference in number of attached hydrogens of each heavy atom
    t1_h_counts = mol_start.get_neighbour_counts('H')
    t2_h_counts = {reverse_mapping[atom_id]: count
                   for atom_id, count in mol_end.get_neighbour_counts('H').items()
                   if mol_end.get_atom(atom_id).element != 'H'}
    delta_h_counts = {atom_id: t2_h_counts[atom_id] - t1_h_counts[atom_id] for atom_id in t2_h_counts}

    # get heavy atoms and their bonds
    t1_heavy_atoms = mol_start.get_heavy_atoms()
    t1_heavy_atom_bonds = mol_start.get_heavy_atom_bonds()
    t2_heavy_atoms = mol_end.get_heavy_atoms()
    t2_heavy_atom_bonds = mol_end.get_heavy_atom_bonds()

    t1_formal_charges = mol_start.get_formal_charges(t1_heavy_atoms)
    t2_formal_charges = {reverse_mapping[atom_id]: formal_charge
                         for atom_id, formal_charge in mol_end.get_formal_charges(t2_heavy_atoms).items()}
    delta_formal_charges = {atom_id: t2_formal_charges[atom_id] - t1_formal_charges[atom_id]
                            for atom_id in t1_formal_charges.keys()}

    t1_bond_orders = mol_start.get_bond_orders(t1_heavy_atom_bonds)
    t2_bond_orders = {frozenset((reverse_mapping[list(bond_id)[0]], reverse_mapping[list(bond_id)[1]])): order
                      for bond_id, order in mol_end.get_bond_orders(t2_heavy_atom_bonds).items()}
    delta_bond_orders = {bond_id: t2_bond_orders[bond_id] - t1_bond_orders[bond_id]
                         for bond_id in t1_bond_orders.keys()}

    print("t1_formal_charges: ", t1_formal_charges)
    print("t2_formal_charges: ", t2_formal_charges)
    print("t2 unmapped charges: ", mol_end.get_formal_charges(t2_heavy_atoms))
    print("Delta formal charges: ", delta_formal_charges)
    assert t1_formal_charges.keys() == t2_formal_charges.keys()

    print("t1_bond_orders: ", t1_bond_orders)
    print("t2_bond_orders: ", t2_bond_orders)
    print("t2 unmapped bond orders: ", mol_end.get_bond_orders(t2_heavy_atom_bonds))
    print("Delta bond_orders: ", delta_bond_orders)

    print(t1_bond_orders.keys())
    print(t2_bond_orders.keys())
    assert t1_bond_orders.keys() == t2_bond_orders.keys()

    print("t1_h_counts: ", t1_h_counts)
    print("t2_h_counts: ", t2_h_counts)
    print("Delta h counts: ", delta_h_counts)

    # 4.
    # create new graph with atom/edge labels from G1, with attributes as the deltas determined above
    # based on number of hydrogens difference at each heavy atom, add newly indexed hydrogens to
    # heavy atoms, with edge labels as -1 if removed, 0 if the same and +1 if added

    # 5. DONE! Return graph! (or new molecule object with this graph..?)
    # after this you need to weave featurise this graph... lol

    # get heavy atoms for trans mol
    trans_atoms = [
        Atom2D(
            index={'name': atom_id},
            name=atom_id,
            element=mol_start.get_atom(atom_id).element,
            formal_charge=delta_formal_charges[atom_id],
            #non_bonded_electrons=t2_non[atom_id] - t1_formal_charges[atom_id],
        )
        for atom_id in t1_heavy_atoms
    ]

    # add hydrogens with new ids
    trans_atoms.extend([
        Atom2D(
            index={'name': f"H{i}"},
            name=f"H{i}",
            element='H',
            formal_charge=0,
        )
        for i in range(1, mol_start.get_element_count('H') + 1)
    ])

    # add heavy atom bonds with delta bond orders
    trans_bonds = []
    for bond_id in t1_bond_orders.keys():
        trans_bond = Bond2D(order=delta_bond_orders[bond_id])
        a1_ind, a2_ind = list(bond_id)
        trans_bonds.append((a1_ind, a2_ind, trans_bond))

    # add bonds to hydrogens with delta bond orders
    h_id = 0
    for atom_id in t1_heavy_atoms:
        pass
        # TODO: need a nice way to add hydrogens back onto mol with delta bond orders

    trans_mol = Molecule2D(
        trans_atoms,
        trans_bonds,
        name="transition",
    )

    return trans_mol


if __name__ == "__main__":
    # DEBUG ONLY
    mol_1 = Molecule2D()
    print()
