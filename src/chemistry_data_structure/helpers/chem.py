from math import pi

LINEAR = 2
TRIGONAL_PLANAR = 3
TETRAHEDRAL = 4
TETRAHEDRAL_BOND_ANGLE = 109.5 * (pi / 180)
TRIGONAL_PLANAR_BOND_ANGLE = 120 * (pi / 180)

VALENCE_ELECTRONS = {
    'H': 1, 'HE': 2,
    'LI': 1, 'BE': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'NE': 8,
    'NA': 1, 'MG': 2, 'AL': 3, 'SI': 4, 'P': 5, 'S': 6, 'CL': 7, 'AR': 8,
    'K': 1, 'CA': 2, 'GA': 3, 'GE': 4, 'AS': 5, 'SE': 6, 'BR': 7, 'KR': 8,
    'RB': 1, 'SR': 2, 'IN': 3, 'SN': 4, 'SB': 5, 'TE': 6, 'I': 7, 'XE': 8,
}

FULL_VALENCES = {
    'C': {4},
    'N': {3, 5},
    'O': {2},
    'H': {1},
    'S': {2, 4, 6},
    'SE': {2, 4, 6},
    'P': {3, 5},
    'CL': {1},
    'BR': {1},
    'F': {1},
    'I': {1, 3, 5},
    'B': {3, 5},
    'SI': {4},
}

ELECTRONEGATIVITIES = {
    # Source: https://en.wikipedia.org/wiki/Electronegativity
    'H': 2.20, 'HE': None,
    'LI': 0.98, 'BE': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'NE': None,
    'NA': 0.93, 'MG': 1.31, 'AL': 1.61, 'SI': 1.90, 'P': 2.19, 'S': 2.58, 'CL': 3.16, 'AR': None,
    'K': 0.82, 'CA': 1.00, 'GA': 1.81, 'GE': 2.01, 'AS': 2.18, 'SE': 2.55, 'BR': 2.96, 'KR': 3.00,
    'RB': 0.82, 'SR': 0.95, 'IN': 1.78, 'SN': 1.96, 'SB': 2.05, 'TE': 2.10, 'I': 2.66, 'XE': 2.60,
}

# unit conversions
BOHR_PER_ANG = 1.8897259885789  # taken straight from the field_fit_interface
BOHR_PER_NM = BOHR_PER_ANG * 10.
AROMATIC_BOND_ORDER = 1.5
