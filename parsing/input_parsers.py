"""
Parsers for the creation of MolecularEntity objects and their derivatives.
"""

from functools import reduce
import re
from io import StringIO

import numpy as np
from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.objects.atom_bond import Atom3D, Bond3D
from chemistry_data_structure.parsing.pdb import bonds_for_pdb_line, is_pdb_connect_line, pdb_atoms_in
from chemistry_data_structure.helpers.chem import BOHR_PER_ANG, BOHR_PER_NM


############# mol2 Parser


def _atom_for_atom_line(line: str):
    index_str, name_str, x, y, z, sybil_atom_type, _, _, partial_charge = line.split()
    element, *_ = sybil_atom_type.split('.')

    return (
        Atom3D(
            index={'mol2': int(index_str)},  # currently an arbitrary dictionary
            name=f'{element}{index_str}',
            element=element,
            valence=None,
            coordinates=(float(x), float(y), float(z))
            # capped=True,
        ),
        float(partial_charge),
    )


AROMATIC_BOND, AMIDE_BOND = 'ar', 'am'


def _bond_for_atom_line(line: str):
    bond_label, atom_id_1, atom_id_2, bond_order_str = line.split()
    if bond_order_str == AROMATIC_BOND:
        bond_order = 1.5
    elif bond_order_str == AMIDE_BOND:
        bond_order = 1
    else:
        bond_order = int(bond_order_str)
    return ([int(atom_id_1), int(atom_id_2)], bond_order)


def mol2_to_Molecule3D(mol2_str: str) -> Molecule3D:
    """
    Primarily adapted from fragment_capping/helpers/molecule.py
    :param mol2_str:
    :return:
    """
    assert mol2_str.count('@<TRIPOS>MOLECULE'), 'Error: MOL2 file does not start with "@<TRIPOS>MOLECULE"'
    assert mol2_str.count('@<TRIPOS>MOLECULE') == 1, 'Only one molecule at a time'

    read_lines, atoms, bonds = False, [], []
    for (i, line) in enumerate(mol2_str.splitlines()):
        if i == 1:
            molecule_name = line
        elif line.startswith('@<TRIPOS>ATOM'):
            container, line_reading_fct, read_lines = atoms, _atom_for_atom_line, True
        elif line.startswith('@<TRIPOS>BOND'):
            container, line_reading_fct, read_lines = bonds, _bond_for_atom_line, True
        elif line.startswith('@'):
            read_lines = False
        else:
            if read_lines:
                container.append(line_reading_fct(line))

    bond_objects = []
    for b in bonds:
        (mol2_id1, mol2_id2), bond_order = b
        for a in atoms:
            a1_name = [a.name for (a, _) in atoms if a.get_index('mol2') == mol2_id1][0]
            a2_name = [a.name for (a, _) in atoms if a.get_index('mol2') == mol2_id2][0]
            bond_objects.append((a1_name, a2_name, Bond3D()))

    total_net_charge = sum(partial_charge for (atom, partial_charge) in atoms)
    assert abs(total_net_charge - round(total_net_charge)) <= 0.01, total_net_charge

    return Molecule3D(
        [atom for (atom, _) in atoms],
        bond_objects,
        # formal_charges={atom.index: round(partial_charge) for (atom, partial_charge) in atoms},
        # bond_orders={bond: bond_order for (bond, bond_order) in bonds},
        name=molecule_name,
        # net_charge=round(total_net_charge),
    )


def pdb_to_Molecule3D(pdb_str: str,
                      mol_name: str = "",
                      net_charge: int = None,
                      assign_bond_orders_and_charges: bool = False) -> Molecule3D:
    """
    Create a 3D molecular entity from a PDB file.
    :param pdb_str:
    :param net_charge:
    :return:
    """

    # TODO: assert statements

    # get pdb atoms
    pdb_atoms = [pdb_atom for pdb_atom in pdb_atoms_in(pdb_str)]

    # convert to chem_ds atoms
    atoms = []
    n_id = 0
    for pdb_atom in pdb_atoms:
        atoms.append(
            Atom3D(
                index={'pdb': int(pdb_atom.index), 'nid': n_id},
                name=pdb_atom.name,
                element=pdb_atom.element,
                coordinates=pdb_atom.coordinates
            )
        )
        n_id += 1

    # get pdb bonds
    pdb_bonds = reduce(
        lambda acc, e: acc | e,
        [
            bonds_for_pdb_line(line)
            for line in pdb_str.splitlines()
            if is_pdb_connect_line(line)
        ],
        set(),
    )

    # print("PDB Bonds: ", pdb_bonds)

    # convert pdb_bonds to chem_ds bonds
    pdb_atom_index_name_map = {pdb_atom.index: pdb_atom.name for pdb_atom in pdb_atoms}
    bonds = []
    for pdb_bond in pdb_bonds:
        a1_ind, a2_ind = list(pdb_bond)
        bonds.append((pdb_atom_index_name_map[a1_ind], pdb_atom_index_name_map[a2_ind], Bond3D()))

    # print("Bonds: ", bonds)

    molecule = Molecule3D(
        atoms,
        bonds,
        name=mol_name
    )

    # assign bond orders and charges with ILP
    if assign_bond_orders_and_charges and net_charge is not None:
        molecule.assign_bond_orders_and_charges_with_ILP(net_charge)

    # if assign aromatic bonds
    # molecule.assign_aromatic_bonds() # TODO: implement this!!!

    return molecule


class BlockException(Exception):
    pass


def GAMESS_to_Molecule3D(
        GAMESS_log: str,
        mol_name: str = '',
        units='Bohr') -> Molecule3D:
    """
    Function for generating 3d molecules from GAMESS qm logs
    Parser, mostly copied from fieldfit interface, but with significant speedups
    :param units: Bohr or Angs (Angstrom)
    :param GAMESS_log: string of the gamess log being parsed
    :param mol_name: name of the molecule
    :return: Molecule3D object with the information from the log
    """
    # todo gamess log has valence information
    # todo it willl be worth investing in the most efficient way to parse the esp field into numerical data

    if units == 'Bohr':
        coord_unit_conversion = BOHR_PER_ANG
    elif units == 'Angs':
        coord_unit_conversion = 1
    else:
        raise Exception('Unrocognised units')

    # this locates the equilibrium geometry block
    # need to escape asterixes and newlines in regex
    # ATOM_BLOCK_HEADING = r"      \*\*\*\*\* EQUILIBRIUM GEOMETRY LOCATED \*\*\*\*\*\n COORDINATES OF ALL ATOMS ARE \(ANGS\)\n   ATOM   CHARGE       X              Y              Z\n ------------------------------------------------------------\n"

    # `Y{x}` matches Y, x times, y can be a space
    ATOM_BLOCK_HEADING = r" {6}\*{5} EQUILIBRIUM GEOMETRY LOCATED \*{5}\n COORDINATES OF ALL ATOMS ARE \(ANGS\)\n   ATOM   CHARGE {7}X {14}Y {14}Z\n -{60}\n"

    # NEXT_BLOCK_HEADING = r"          INTERNUCLEAR DISTANCES \(ANGS\.\)\n          ------------------------------"
    NEXT_BLOCK_HEADING = r" {10}INTERNUCLEAR DISTANCES \(ANGS\.\)\n {10}-{30}"

    # units are in angstroms
    # todo need to check if there is some method for tracking this

    compile_str = f"(?<={ATOM_BLOCK_HEADING})[\\s\\S]+?(?={NEXT_BLOCK_HEADING})"
    parser = re.compile(compile_str)
    atom_result = parser.findall(GAMESS_log)
    if not atom_result:
        raise BlockException("Equilibrium Atom Block not found")

    # index for gamess are done in the order they appear in the log

    # fetching the bond/valence block
    BOND_BLOCK_HEADING = r" {19}BOND {23}BOND {23}BOND\n  ATOM PAIR DIST  ORDER      ATOM PAIR DIST  ORDER      ATOM PAIR DIST  ORDER"

    # VALENCE_BLOCK_HEADING = r"\n                       TOTAL       BONDED        FREE\n      ATOM            VALENCE     VALENCE     VALENCE"
    VALENCE_BLOCK_HEADING = r" {23}TOTAL       BONDED        FREE\n {6}ATOM {12}VALENCE     VALENCE     VALENCE"

    # NEXT_BLOCK_HEADING = r"\n          ---------------------\n          ELECTROSTATIC MOMENTS\n          ---------------------"
    NEXT_BLOCK_HEADING = r" {10}-{21}\n {10}ELECTROSTATIC MOMENTS\n {10}-{21}"

    compile_str = f"(?<={BOND_BLOCK_HEADING})[\\s\\S]+?(?={VALENCE_BLOCK_HEADING})"
    parser = re.compile(compile_str)
    bond_result = parser.findall(GAMESS_log)
    if not bond_result:
        raise BlockException("Equilibrium Bond Block not found")

    compile_str = f"(?<={VALENCE_BLOCK_HEADING})[\\s\\S]+?(?={NEXT_BLOCK_HEADING})"
    parser = re.compile(compile_str)
    valence_result = parser.findall(GAMESS_log)
    if not bond_result:
        raise BlockException("Valency Bond Block not found")
    valencies = {}
    for line in valence_result[0].strip('\n').split('\n'):
        index, element, tot_val, bond_val, free_val = line.split()
        valencies[int(index)] = float(tot_val)

    # parsing the qm esp grid (units of BOHRs)
    # the esp grid has some arbitrary comments/headings in odd places making the regex a bit complex
    # requires putting the relevent data in group one and extracting it as such
    compile_str = r"(?<=ELECTROSTATIC POTENTIAL)[\s\S]+?= *\d*\n([\s\S]+?)(?=\n NET CHARGES:)"
    parser = re.compile(compile_str)
    optimised_grid = parser.findall(GAMESS_log)[-1]
    with StringIO(optimised_grid) as str_buffer:
        # this seems to be one of the fastest ways to load the esp_grid
        esp_matrix = np.loadtxt(str_buffer, dtype=np.float64)
    # default for these is BOHR
    esp_grid_coords = esp_matrix.T[[1, 2, 3]].T  # want each row to be xyz
    esp_grid_charges = esp_matrix.T[6]
    atoms = {}
    for index, line in enumerate(atom_result[0].strip('\n').split('\n')):
        index += 1  # indexes start at 1
        element, atomic_charge, x, y, z = line.split()
        atom_name = f'{element}{index}'  # TODO need to check if there are underscores between these
        atoms[index] = Atom3D(
            name=atom_name,
            element=element,
            coordinates=[float(c) * coord_unit_conversion for c in [x, y, z]], # convert from angstrom
            index={'index': index},
            formal_charge=float(atomic_charge),  # todo need to check if these are the right charges, also not wokring
            valence=valencies[index]

        )

    bonds = []
    for line in bond_result[0].strip('\n').split('\n'):
        # up to 3 groups per line
        groups = line.split('        ')
        for g in groups:
            id1, id2, distance, bond_order = g.split()
            atom1_name = atoms[int(id1)].name
            atom2_name = atoms[int(id2)].name

            bonds.append((
                atom1_name,
                atom2_name,
                Bond3D(order=bond_order)
            )
            )

    return Molecule3D(atoms=list(atoms.values()),
                      bonds=bonds,
                      esp_grid_coords = esp_grid_coords,
                      esp_grid_charge = esp_grid_charges
                      )


if __name__ == '__main__':
    # with open('data/benxene.mol2.txt', 'r') as f:
    #     test = mol2_to_Molecule3D(f.read())
    #
    #     import networkx as nx
    #
    #     nx.set_node_attributes(test.graph, 'test', 'test')
    #     nx.get_node_attributes(test.graph, 'test')

    with open('../test/data/qm/451_b3lyp_631Gd.out', 'r') as f:
        test2 = GAMESS_to_Molecule3D(f.read(), units='Bohr')
