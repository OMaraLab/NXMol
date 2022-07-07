"""
Contains helper methods for parsing pdb files.
"""

from typing import List, FrozenSet, Optional, NamedTuple, Tuple, Any

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


def is_pdb_atom_line(line: str) -> bool:
    return line[0:len(PDB_ATOM_RECORDS[0])] in PDB_ATOM_RECORDS


def is_pdb_connect_line(line: str) -> bool:
    return line[0:len(PDB_CONNECT_RECORDS[0])] in PDB_CONNECT_RECORDS


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
