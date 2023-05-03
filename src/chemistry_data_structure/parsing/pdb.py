"""
Contains helper methods for parsing pdb files.
"""

from typing import List, FrozenSet, Optional, NamedTuple, Tuple, Any, Union

Coordinate = Tuple[float, float, float]

# Columns definition are inclusive on both ends
PDB_ATOM_INDEX_FIELD = (7, 11)
PDB_ATOM_NAME_FIELD = (12, 16)
PDB_COORD_FIELDS = ((31,38), (39, 46), (47, 54)) # Taken from ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_Letter.pdf, page 194
PDB_ELEMENT_FIELD = (77, 78)
PDB_CHARGE = (79, 80)

PDB_INT = int
PDB_STR = lambda x: str(x).strip()
def PDB_MAYBE_INT_WITH_SIGN(x: str) -> int:
    if x.strip() == '':
        return None
    else:
        charge_int, charge_sign = x
        if charge_sign == '-':
            sign = -1
        elif charge_sign == '+':
            sign = +1
        else:
            raise Exception('Unexpected charge_sign: "{0}" (x="{1}")'.format(charge_sign, x))
        return sign * int(charge_int)

PDB_FIELD_AND_FORMATTER_FOR_ATTRIBUTE = {
    'atom_index': (PDB_ATOM_INDEX_FIELD, PDB_INT),
    'atom_name': (PDB_ATOM_NAME_FIELD, PDB_STR),
    'element': (PDB_ELEMENT_FIELD, PDB_STR),
    'charge': (PDB_CHARGE, PDB_MAYBE_INT_WITH_SIGN),
}

PDB_ATOM_RECORDS = ('ATOM  ', 'HETATM')
PDB_CONNECT_RECORDS = ('CONECT',)

PDB_Atom = NamedTuple(
    'PDB_Atom',
    [
        ('index', int),
        ('name', str),
        ('coordinates', Tuple[float, float, float]),
        ('element', str),
        ('charge', Optional[int]),
    ]
)

S = ('>', '')
CHARGE = ('<', '')
F = ('', '.3f')
I = ('>', '')

# Taken from: ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf, p183
HETATM_SPECS = (
    (1, 6) + S,   # Record name
    PDB_ATOM_INDEX_FIELD + S,  # Atom serial number
    PDB_ATOM_NAME_FIELD + S, # Atom name
#    (17, 17) + S, # Alternate location indicator
    (18, 21) + S, # Residue name
    (22, 22) + S, # Chain identifier
    (23, 26) + S, # Residue sequence number
#    (27, 27) + S, # Code for insertion of residues
    PDB_COORD_FIELDS[0] + F, # Orthogonal coordinates for X
    PDB_COORD_FIELDS[1] + F, # Orthogonal coordinates for Y
    PDB_COORD_FIELDS[2] + F, # Orthogonal coordinates for Z
    (55, 60) + S, # Occupancy
    (61, 66) + S, # Temperature factor
    PDB_ELEMENT_FIELD + S, # Element symbol; right-justified
    PDB_CHARGE + CHARGE, # Charge on the atom
)

CONECT_SPECS = (
    (1, 6) + S, # Record name
    (7, 11) + I, # Atom serial number
    (12, 16) + I, # Serial number of bonded atom
    (17, 21) + I, # Serial number of bonded atom
    (22, 26) + I, # Serial number of bonded atom
    (27, 31) + I, # Serial number of bonded atom
)


def is_pdb_atom_line(line: str) -> bool:
    return line[0:len(PDB_ATOM_RECORDS[0])] in PDB_ATOM_RECORDS


def is_pdb_connect_line(line: str) -> bool:
    return line[0:len(PDB_CONNECT_RECORDS[0])] in PDB_CONNECT_RECORDS

def pdb_conect_line(fields: List[Union[str, int]]) -> List[str]:
    return CONECT_TEMPLATE.format(
        *list(PDB_CONNECT_RECORDS) + fields + [''] * (len(CONECT_SPECS) - len(fields) - 1)
    )

def pdb_atoms_in(pdb_str: str) -> List[PDB_Atom]:
    return [
        PDB_Atom(
            index=get_attribute_from_pdb_line('atom_index', line),
            name=get_attribute_from_pdb_line('atom_name', line),
            coordinates=get_coords_from_pdbline(line),
            element=get_attribute_from_pdb_line('element', line),
            charge=get_attribute_from_pdb_line('charge', line),
        )
        for line in pdb_str.splitlines()
        if is_pdb_atom_line(line)
    ]


def get_coords_from_pdbline(line: str) -> Optional[Coordinate]:
    if is_pdb_atom_line(line):
        return tuple(
            map(
                float,
                [
                    line[pdb_coord_field[0] - 1: pdb_coord_field[1]]
                    for pdb_coord_field in PDB_COORD_FIELDS
                ],
            ),
        )
    else:
        return None


def bonds_for_pdb_line(pdb_line: str) -> List[FrozenSet[int]]:
    atom_index, *other_atom_indices = map(int, pdb_line.split()[1:])
    return {
        frozenset((atom_index, other_atom_index))
        for other_atom_index in other_atom_indices
    }


def get_attribute_from_pdb_line(attribute: str, line: str) -> Any:
    pdb_field, formatter = PDB_FIELD_AND_FORMATTER_FOR_ATTRIBUTE[attribute]
    return formatter(line[pdb_field[0] - 1: pdb_field[1]])

def PDB_FORMAT_STR(pdb_specs):
    all_fields = []

    for i, fields in enumerate(pdb_specs):
        if i == 0:
            diff = 0
        else:
            diff = (fields[0] -1) - (pdb_specs[i - 1][1])
        if diff != 0:
            all_fields.append(' ' * diff)

        all_fields.append(
            '{' + '{i}:{left_or_right}{len}{formatter}'.format(
            i=i,
            len=(fields[1] - fields[0] + 1),
            left_or_right=fields[2],
            formatter=fields[3],
        ) + '}'
        )
    return ''.join(all_fields)

PDB_TEMPLATE = PDB_FORMAT_STR(HETATM_SPECS)
CONECT_TEMPLATE = PDB_FORMAT_STR(CONECT_SPECS)
