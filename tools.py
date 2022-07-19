"""
Contains methods for performing transformations on molecular entities etc.
"""
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

    # 1. get a mapping between start and end structure

    def nodes_equal(n1, n2):
        return n1.element == n2.element

    # ISMAGS implementation
    G1 = mol_start.graph
    G2 = mol_end.graph
    ismags = isomorphism.ISMAGS(G1, G2, node_match=nodes_equal)
    print("Is isomorphic? ", ismags.is_isomorphic())
    largest_common_subgraphs = list(ismags.largest_common_subgraph())
    print("Largest common subgraphs: ", largest_common_subgraphs)
    g1_g2_mapping = largest_common_subgraphs[0]
    print("Start -> End Structure Mapping: ", g1_g2_mapping)

    # TODO: WE CAN USE THIS, OR VF2 ISOMORPHISM, BUT EITHER WAY NEED TO STRIP HYDROGENS FIRST
    #   AS OTHERWISE THERE ARE COMPUTATIONAL COMPLEXITY PROBLEMS

    # 2. Using this mapping, create a new molecule object with attributes representing
    # the difference in bond_orders/formal_charges etc.


    # TODO:
    #  effectively need to create new Molecule2D, with subtracted bond orders of bonds,
    #  subtracted formal charges of atoms, ensuring that the same atom/bond ids are used for this process

    trans_mol = Molecule2D()


    return Molecule2D()


if __name__ == "__main__":
    # DEBUG ONLY
    mol_1 = Molecule2D()
    print()
