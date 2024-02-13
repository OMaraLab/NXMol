"""
Parsers for the creation of MolecularEntity objects and their derivatives.
"""

from functools import reduce
import re
from io import StringIO

import networkx as nx
import numpy as np
import pickle
from pprint import pprint
from chemistry_data_structure.parsing.hessian_analysis import (
    cal_eigen_matrix,
    cal_stretching,
)
from chemistry_data_structure.objects.atom_bond import Atom3D, Bond3D, Bond2D
from chemistry_data_structure.objects.molecular_entity import (
    Molecule3D,
)  # TODO: this import has broken??
from chemistry_data_structure.parsing.pdb import (
    bonds_for_pdb_line,
    is_pdb_connect_line,
    pdb_atoms_in,
)
from chemistry_data_structure.helpers.chem import (
    BOHR_PER_ANG,
    BOHR_PER_NM,
    FULL_VALENCES,
    VALENCE_ELECTRONS,
)


def _atom_for_atom_line(line: str):
    index_str, name_str, x, y, z, sybil_atom_type, _, _, partial_charge = line.split()
    element, *_ = sybil_atom_type.split(".")

    return (
        Atom3D(
            index={"mol2": int(index_str)},  # currently an arbitrary dictionary
            name=f"{element}{index_str}",
            element=element,
            valence=None,
            coordinates=(float(x), float(y), float(z)),
        ),
        float(partial_charge),
    )


AROMATIC_BOND, AMIDE_BOND = "ar", "am"


def _bond_for_atom_line(line: str):
    bond_label, atom_id_1, atom_id_2, bond_order_str = line.split()
    if bond_order_str == AROMATIC_BOND:
        bond_order = 1.5
    elif bond_order_str == AMIDE_BOND:
        bond_order = 1
    else:
        bond_order = int(bond_order_str)
    return ([int(atom_id_1), int(atom_id_2)], bond_order)


def pickle_to_Molecule3D(molid: str):
    """
    Parse a pickled QM output file to construct a Molecule3D
    """
    qm_data = pickle.load(
        open(f"test_dataset/{molid}/b3lyp_631Gd_PCM_water_hessian.pickle", "rb")
    )

    atoms = [
        Atom3D(x, x, tuple(y))
        for x, y in zip(
            qm_data["type"].values(), qm_data["primary_axis_coords"].values()
        )
    ]

    fc = {}
    umatrix, eigmatrix = cal_eigen_matrix(
        qm_data["primary_axis_coords"], qm_data["hessian"]
    )

    for i, j, bond_order in qm_data["bond_order"]:
        if bond_order > 0.85:
            fc[frozenset([i, j])] = cal_stretching([i, j], umatrix, eigmatrix)

    bonds = [
        (i, j, fc[frozenset([i, j])])
        for i, j, bond_order in qm_data["bond_order"]
        if bond_order > 0.85
    ]

    troll = Molecule3D(atoms=atoms, bonds=[x for x in bonds], name=molid)
    print(troll.formal_charges)
    pass


def mol2_to_Molecule3D(mol2_str: str) -> Molecule3D:
    """
    Primarily adapted from fragment_capping/helpers/molecule.py
    :param mol2_str:
    :return:
    """
    assert mol2_str.count(
        "@<TRIPOS>MOLECULE"
    ), 'Error: MOL2 file does not start with "@<TRIPOS>MOLECULE"'
    assert mol2_str.count("@<TRIPOS>MOLECULE") == 1, "Only one molecule at a time"

    read_lines, atoms, bonds = False, [], []
    for i, line in enumerate(mol2_str.splitlines()):
        if i == 1:
            molecule_name = line
        elif line.startswith("@<TRIPOS>ATOM"):
            container, line_reading_fct, read_lines = atoms, _atom_for_atom_line, True
        elif line.startswith("@<TRIPOS>BOND"):
            container, line_reading_fct, read_lines = bonds, _bond_for_atom_line, True
        elif line.startswith("@"):
            read_lines = False
        else:
            if read_lines:
                container.append(line_reading_fct(line))

    bond_objects = []
    for b in bonds:
        (mol2_id1, mol2_id2), bond_order = b
        for a in atoms:
            a1_name = [a.name for (a, _) in atoms if a.get_index("mol2") == mol2_id1][0]
            a2_name = [a.name for (a, _) in atoms if a.get_index("mol2") == mol2_id2][0]
            bond_objects.append((a1_name, a2_name, Bond3D(set((a1_name, a2_name)))))

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


def pdb_index_parser(pdb_str: str):
    template_exp = "(?<=(HETATM|ATOM  ))(.*)"
    # so far these are the only lines we care about exporting
    parser = re.compile(template_exp)
    atm_lines = parser.findall(pdb_str)
    id_map = {}
    for line in atm_lines:
        split_line = line[1].strip().split()
        GAMESS_id = split_line[0]
        atom_name = split_line[1]
        id_map[int(GAMESS_id)] = atom_name
    return id_map


def pdb_to_Molecule3D(
    pdb_str: str,
    mol_name: str = "",
    net_charge: int = None,
    assign_bond_orders_and_charges: bool = False,
) -> Molecule3D:
    """
    Create a 3D molecular entity from a PDB file.
    :param pdb_str: a string of the pdb file contents
    :param net_charge: the net charge of the input molecule
    :return: Molecule3D as read from the input pdb.
    """

    # TODO: assert statements

    # get pdb atoms
    pdb_atoms = [pdb_atom for pdb_atom in pdb_atoms_in(pdb_str)]

    # convert to chem_ds atoms
    atoms = []
    for n_id, pdb_atom in enumerate(pdb_atoms, start=1):
        atoms.append(
            Atom3D(
                index={"pdb": int(pdb_atom.index), "nid": n_id},
                name=pdb_atom.name.replace("_", ""),
                element=pdb_atom.element,
                coordinates=pdb_atom.coordinates,
                full_valence=FULL_VALENCES[pdb_atom.element.upper()],
                valence_electrons=VALENCE_ELECTRONS[pdb_atom.element.upper()],
            )
        )

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

    # convert pdb_bonds to chem_ds bonds
    pdb_atom_index_name_map = {
        pdb_atom.index: pdb_atom.name.replace("_", "") for pdb_atom in pdb_atoms
    }
    bonds = []
    for pdb_bond in pdb_bonds:
        a1_ind, a2_ind = list(pdb_bond)
        a1_name, a2_name = (
            pdb_atom_index_name_map[a1_ind],
            pdb_atom_index_name_map[a2_ind],
        )
        bonds.append((a1_name, a2_name, Bond3D(set((a1_name, a2_name)))))

    molecule = Molecule3D(atoms, bonds, name=mol_name, net_charge=net_charge)

    if assign_bond_orders_and_charges and net_charge is not None:
        # assign bond orders and charges with ILP
        molecule.assign_bond_orders_and_charges_with_ILP(net_charge=net_charge)

        # assign aromatic bonds, hybridisations, actual valences and conjugations
        molecule.assign_aromatic_bonds()
        molecule.assign_hybridisations_and_valences()
        molecule.assign_conjugated_atoms()

    # print("Mol Name: ", mol_name)
    # print("Atoms: ", molecule.atoms)
    # print("Valences: ", molecule.valences)
    # print("Non-bonded Electrons: ", molecule.non_bonded_electrons)
    # print("Hybridisations: ", molecule.hybridisations)
    # print("Conjugations", molecule.atom_conjugations)
    # print("Formal Charges: ", molecule.formal_charges)
    # print("Bonds: ", molecule.bonds)
    # print("Bond Orders: ", molecule.bond_orders)

    return molecule


class BlockException(Exception):
    pass


def _GAMESS_parser(GAMESS_log: str, units: str = "Bohr", id_map=None):
    if units == "Bohr":
        coord_unit_conversion = BOHR_PER_ANG
    elif units == "Angs":
        coord_unit_conversion = 1
    else:
        raise Exception("Unrecognised units")
    # this locates the equilibrium geometry block
    # need to escape asterixes and newlines in regex
    # ATOM_BLOCK_HEADING = r"      \*\*\*\*\* EQUILIBRIUM GEOMETRY LOCATED \*\*\*\*\*\n COORDINATES OF ALL ATOMS ARE \(ANGS\)\n   ATOM   CHARGE       X              Y              Z\n ------------------------------------------------------------\n"
    # `Y{x}` matches Y, x times, Y can be a space

    if id_map is not None:
        mapping = lambda x: id_map[x]
    else:
        mapping = lambda x: x

    ATOM_BLOCK_HEADING = r" {6}\*{5} EQUILIBRIUM GEOMETRY LOCATED \*{5}\n COORDINATES OF ALL ATOMS ARE \(ANGS\)\n   ATOM   CHARGE {7}X {14}Y {14}Z\n -{60}\n"
    NEXT_BLOCK_HEADING = (
        r"\n\n"  # search for the first empty line after the coordinates block
    )
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
    # NEXT_BLOCK_HEADING = r" {10}-{21}\n {10}ELECTROSTATIC MOMENTS\n {10}-{21}"
    # ALT_comment = r"    \*\*\*\* A SOLVENT MODEL IS IN USE IN THIS RUN \*\*\*\*"
    compile_str = f"(?<={BOND_BLOCK_HEADING})[\\s\\S]+?(?={VALENCE_BLOCK_HEADING})"
    parser = re.compile(compile_str)
    bond_result = parser.findall(GAMESS_log)
    if not bond_result:
        raise BlockException("Equilibrium Bond Block not found")
    # compile_str = f"(?<={VALENCE_BLOCK_HEADING})[\\s\\S]+?(?={NEXT_BLOCK_HEADING}|{ALT_comment})"
    compile_str = f"(?<={VALENCE_BLOCK_HEADING})[\\s\\S]+?(?=\n\n)"
    parser = re.compile(compile_str)
    valence_result = parser.findall(GAMESS_log)
    if not valence_result:
        raise BlockException("Valency Bond Block not found")
    valencies = {}
    for line in valence_result[0].strip("\n").split("\n"):
        index, element, tot_val, bond_val, free_val = line.split()
        valencies[int(index)] = float(tot_val)
    # parsing the qm esp grid (units of BOHRs)
    # the esp grid has some arbitrary comments/headings in odd places making the regex a bit complex
    # requires putting the relevent data in group one and extracting it as such
    compile_str_ESP_with_commments = (
        r"(?<=ELECTROSTATIC POTENTIAL)([\s\S]+?)(?=\n NET CHARGES:)"
    )
    parser = re.compile(compile_str_ESP_with_commments)
    grid_with_heading = parser.findall(GAMESS_log)[-1]
    compile_str_extract_grid_start = r"\s*\d*(\s*-?\d*\.?\d+){6}\n"
    # parser = re.compile(compile_str_extract_grid_start)
    start_index = re.search(compile_str_extract_grid_start, grid_with_heading).start()
    optimised_grid = grid_with_heading[start_index:]

    with StringIO(optimised_grid) as str_buffer:
        # this seems to be one of the fastest ways to load the esp_grid
        esp_matrix = np.loadtxt(str_buffer, dtype=np.float64)
    # default for these is BOHR
    esp_grid_coords = esp_matrix.T[[1, 2, 3]].T  # want each row to be xyz
    esp_grid_charges = esp_matrix.T[6]
    atoms = {}
    for index, line in enumerate(atom_result[0].strip("\n").split("\n")):
        index += 1  # indexes start at 1
        element, atomic_charge, x, y, z = line.split()
        GAMESS_name = f"{element}{index}"
        if id_map is None:
            atom_name = (
                GAMESS_name  # TODO need to check if there are underscores between these
            )
            index_data = {
                "index": index,
                "GAMESS_index": index,
                "GAMESS_name": GAMESS_name,
            }
        else:
            atom_name = mapping(index)
            index_data = {
                "index": index,
                "GAMESS_index": index,
                "GAMESS_name": GAMESS_name,
                "pdb_name": atom_name,
            }
        atoms[index] = Atom3D(
            name=atom_name,
            element=element,
            coordinates=[
                float(c) * coord_unit_conversion for c in [x, y, z]
            ],  # convert from angstrom
            index=index_data,
            formal_charge=float(
                atomic_charge
            ),  # todo need to check if these are the right charges, also not working
            valence=valencies[index],
        )
    bonds = []
    for line in bond_result[0].strip("\n").split("\n"):
        # up to 3 groups per line
        elements = line.split()
        num_groups = len(elements) // 4
        groups = []
        for i in range(num_groups):
            groups.append([elements[jj + i * 4] for jj in range(4)])

        for g in groups:
            id1, id2, distance, bond_order = g
            if id_map is not None:
                atom1_name = mapping(int(id1))
                atom2_name = mapping(int(id2))
            else:
                atom1_name = atoms[int(id1)].name
                atom2_name = atoms[int(id2)].name

            bonds.append(
                (
                    atom1_name,
                    atom2_name,
                    Bond3D(set((atom1_name, atom2_name)), order=float(bond_order)),
                )
            )

    # extracting net charge
    compile_str_net_charge = "(?<=CHARGE OF MOLECULE\s{27}=)\s*-?\d*"
    parser_net_charge = re.compile(compile_str_net_charge)
    net_charge_result = parser_net_charge.findall(GAMESS_log)
    if not net_charge_result:
        raise BlockException("Net charge line not found")
    net_charge = int(net_charge_result[0])

    return atoms, bonds, esp_grid_charges, esp_grid_coords, net_charge


def GAMESS_to_Molecule3D(
    GAMESS_log: str, mol_name: str = "", units="Bohr"
) -> Molecule3D:
    """
    Function for generating 3d molecules from GAMESS qm logs
    Parser, mostly copied from fieldfit interface, but with significant speedups
    :param units: Bohr or Angs (Angstrom)
    :param GAMESS_log: string of the gamess log being parsed
    :param mol_name: name of the molecule
    :return: Molecule3D object with the information from the log
    """
    # todo GAMESS log has valence information
    # todo it will be worth investing in the most efficient way to parse the esp field into numerical data
    # this parser takes ~0.2 seconds might add option to not parse the qm logs
    # mmap may be a solution but there is debate

    atoms, bonds, esp_grid_charges, esp_grid_coords, net_charge = _GAMESS_parser(
        GAMESS_log, units
    )

    return Molecule3D(
        atoms=list(atoms.values()),
        bonds=bonds,
        esp_grid_coords=esp_grid_coords,
        esp_grid_charge=esp_grid_charges,
        net_charge=net_charge,
    )


def GAMESS_pdb_to_Molecule3D(
    pdb_str: str,
    GAMESS_str: str,
):
    """
    This molecule should have been imediately initialised with a GAMMESS Parser
    :param pdb_str:
    :param molecule:
    :return:
    """

    id_map = pdb_index_parser(pdb_str)
    atoms, bonds, esp_grid_charges, esp_grid_coords, net_charge = _GAMESS_parser(
        GAMESS_log=GAMESS_str, id_map=id_map
    )

    return Molecule3D(
        atoms=list(atoms.values()),
        bonds=bonds,
        esp_grid_coords=esp_grid_coords,
        esp_grid_charge=esp_grid_charges,
        net_charge=net_charge,
    )


def mol_to_Molecule3D(mol_str: str):
    """
    Read a molecule from a mol file, does not read in auxiliary information (TODO: in the future add metadata)
    :param mol_str: the mol file str.
    :return: Molecule3D object.
    """

    mol_str_lines = mol_str.split("\n")

    # read header
    mol_name = mol_str_lines[0]

    # read overall mol info
    mol_info = re.findall("[0-9]+", mol_str_lines[3])
    num_atoms = int(mol_info[0])
    num_bonds = int(mol_info[1])

    # dictionary for mol formatting of charges conversion
    mol_charge_read_dict = {7: -3, 6: -2, 5: -1, 0: 0, 3: 1, 2: 2, 1: 3}

    # constants for mol parsing
    ATOM_START_LINE = 4
    atom_end_line = ATOM_START_LINE + num_atoms

    # read atoms
    n_id = 1
    id_name_map = {}
    atoms = []
    for atom_line in mol_str_lines[ATOM_START_LINE:atom_end_line]:
        atom_info = re.findall("[-]?[0-9]+\.[0-9]+|[A-Za-z]+|[0-9]+", atom_line)

        x = float(atom_info[0])
        y = float(atom_info[1])
        z = float(atom_info[2])
        element = atom_info[3]
        formal_charge = mol_charge_read_dict[int(atom_info[5])]
        atom_name = element + str(n_id)

        atom = Atom3D(
            name=atom_name,
            element=element,
            coordinates=[float(c) for c in [x, y, z]],
            index={"name": atom_name, "nid": n_id},
            formal_charge=formal_charge,
        )
        atoms.append(atom)

        id_name_map[n_id] = atom.get_index("name")
        n_id += 1

    # sum up formal charges to find net charge
    net_charge = sum([atom.formal_charge for atom in atoms])

    # write bonds
    bond_start_line = atom_end_line
    bond_end_line = bond_start_line + num_bonds
    bonds = []
    for bond_line in mol_str_lines[bond_start_line:bond_end_line]:
        # get bond info from mol line
        bond_info = re.findall("[0-9]+", bond_line)
        a1_id = int(bond_info[0])
        a2_id = int(bond_info[1])
        a1_name = id_name_map[a1_id]
        a2_name = id_name_map[a2_id]
        order = int(bond_info[2])

        # create bond object
        bond = Bond3D(set((a1_name, a2_name)), order=order)
        bonds.append((a1_name, a2_name, bond))

    return Molecule3D(atoms, bonds, name=mol_name, net_charge=net_charge)


def gml_to_Molecule3D(fpath: str):
    """
    Read a molecule from a GML file.
    TODO: make a 2D version of this, or the option to read a 2D molecule only.
    :param fpath: file path to the mol file.
    :return: Molecule3D object.
    """

    # read graph from gml file
    mol_graph = nx.read_gml(fpath, destringizer=nx.readwrite.gml.literal_destringizer)
    atoms = mol_graph.nodes
    net_charge = mol_graph.graph["net_charge"]

    atoms = []
    for node_dict in mol_graph.nodes.values():
        atom = Atom3D(
            node_dict["index"]["name"], node_dict["element"], node_dict["coordinates"]
        )
        atom.__dict__.update(node_dict)
        atoms.append(atom)

    bonds = []
    for edge_dict in mol_graph.edges.values():
        # fix set formatting of bond
        edge_dict["atoms"] = set(edge_dict["atoms"])

        # make bond object and update attributes dictionary
        bond = Bond3D(edge_dict["atoms"])
        bond.__dict__.update(edge_dict)
        bonds.append(bond)

    bond_list = [
        (list(bond.get_atoms())[0], list(bond.get_atoms())[1], bond) for bond in bonds
    ]
    return Molecule3D(
        atoms=atoms, bonds=bond_list, net_charge=net_charge, name=fpath.split(".")[0]
    )


class BlockException(Exception):
    pass


if __name__ == "__main__":
    # with open('data/benxene.mol2.txt', 'r') as f:
    #     test = mol2_to_Molecule3D(f.read())
    #
    #     import networkx as nx
    #
    #     nx.set_node_attributes(test.graph, 'test', 'test')
    #     nx.get_node_attributes(test.graph, 'test')

    with open("../test/data/qm/451_b3lyp_631Gd.out", "r") as f:
        test2 = GAMESS_to_Molecule3D(f.read(), units="Bohr")
        print(test2.partialChargeFit())
        print(test2.partialChargeFit(solver="pulp"))
        print(test2.partialChargeFit(solver="gurobi", method="round"))
        # print(test2.partialChargeFit(method='ILP', minmax=True))
