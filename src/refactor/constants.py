import enum
#The node feature vector created by get_bond_features() for each pair of atoms in a bond is formatted as follows (verified by looking at get_bond_features()):
class nodeFeatures(enum.IntEnum):
    ATOM1_ID = 0
    ATOM1_ELEMENT = 1
    ATOM1_ATOMIC_NUMBER = 2
    ATOM1_RADIUS = 3
    ATOM1_MASS = 4
    ATOM1_ELECTRONEGATIVITY = 5
    ATOM1_HYBRIDISATION = 6
    ATOM2_ID = 7
    ATOM2_ELEMENT = 8
    ATOM2_ATOMIC_NUMBER = 9
    ATOM2_RADIUS = 10
    ATOM2_MASS = 11
    ATOM2_ELECTRONEGATIVITY = 12
    ATOM2_HYBRIDISATION = 13
    BOND_LENGTH = 14
    BOND_ORDER = 15
    FIRST_DEGREE_NEIGHBOURS = 16

COVALENT_BOND_ORDER_THRESHOLD = 0.5
