"""
Contains methods for performing transformations on molecular entities etc.
"""
from chemistry_data_structure.objects.molecular_entity import Molecule2D


def gen_config_trans_struct_2D(mol_start: Molecule2D,
                               mol_end: Molecule2D) -> Molecule2D:
    """
    Generates a transition molecule representation, from one molecular graph
    to another. This is used to represent configurational isomerism transitions in 2D.
    :param mol_start: the starting Molecule2D
    :param mol_end: the ending Molecule2D
    :return: the transition structure, as a Molecule2D
    """

    # assertions
    # all atoms have formal charge attributes
    # all bonds have bond order attributes


    # TODO:
    #  effectively need to create new Molecule2D, with subtracted bond orders of bonds,
    #  subtracted formal charges of atoms, ensuring that the same atom/bond ids are used for this process

    trans_mol = Molecule2D()


    return Molecule2D()


if __name__ == "__main__":
    # DEBUG ONLY
    mol_1 = Molecule2D()
    print()
