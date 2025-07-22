from collections import defaultdict
import os
import pickle
import re
from statistics import mean
from sys import exit

from chemistry_data_structure.parsing import hessian_analysis, input_parsers
from refactor import featurize, utils


def nested_defaultdict_list():
    return defaultdict(list)


def gather_neighbours(
    csv_path: str,
    data_dir: str = "hessian_data",
    depth: int = 1,
    write: bool = True,
    output_path: str = "gathered_neighbours.pickle",
):
    """
    Gather the {depth}-degree neighbours of every bond in a dataset into a defaultdict(defaultdict(list)), where k : v of the outer dict is bond type (e.g., C:C) : neighbourhood type, and k : v of the inner dict is neighbourhood type : list of bonds with that neighbourhood type. In the list of bonds, each bond is in the format of the return value of Molecule3D.BFS_edge(hybridisation = True).
    NOTE: CURRENTLY ONLY WORKS FOR FIRST DEGREE NEIGHBOURS as BFS_edge is only capable of returning the first degree neighbours.
    """
    gathered_neighbours = defaultdict(nested_defaultdict_list)
    net_charges = utils.load_charges(csv_path)

    for x in utils.progress_bar(
        os.listdir(data_dir), prefix="Gathering neighbours from molecules"
    ):
        qm = featurize.load_qm_data(x)
        try:
            mol3D = input_parsers.ATB_QMData_to_Molecule3D(
                qm, net_charge=net_charges[x], name=x
            )
        except Exception:
            continue

        nei_a = nei_b = bond_tag = None
        for i, j in mol3D.bonds:
            nei_a, nei_b, bond_tag = mol3D.BFS_edge(i, j, depth, hybridisation=True)
            ele_a_id, ele_b_id, _ = tuple(bond_tag.split("_"))
            ele_a = ele_a_id[0]
            ele_b = ele_b_id[0]

            ele_key = tuple(sorted((ele_a, ele_b)))
            nei_key = tuple(sorted((nei_a, nei_b)))

            gathered_neighbours[ele_key][nei_key].append(bond_tag)

    print(
        f"Gathered {len(gathered_neighbours)} element pairs, {len([nei for bond_type in gathered_neighbours for nei in gathered_neighbours[bond_type]])} bond environments, and {sum([len(gathered_neighbours[bond_type][nei]) for bond_type in gathered_neighbours for nei in gathered_neighbours[bond_type]])} bonds from {len(os.listdir(data_dir))} molecules. Written to {output_path}."
    )
    if write:
        pickle.dump(gathered_neighbours, open(output_path, "wb"))
    return gathered_neighbours


def calc_mean_fc(bond_list: list, output_fc_list: bool = False):
    """
    Calculates the mean force constant for a list of bonds. Each bond in the list should be in the format of the return value of Molecule3D.BFS_edge(hybridisation=True), i.e., C1_C2_2001 ({elementa}{atomID}_{element}{atomID}_{molID})
    """
    fc_list = []
    for x in bond_list:
        IDs = re.findall(r"(\d+)", x)
        atoms = re.findall(r"[a-zA-Z]+", x)
        if len(IDs) != 3 or len(atoms) != 2:
            raise ValueError(f"Invalid bond format: {x}")
        qm_data = featurize.load_qm_data(IDs[2])
        qm_pairs = [(a, b) for a, b, _ in qm_data["bond_order"]]
        umatrix, eigmatrix = hessian_analysis.cal_eigen_matrix(
            qm_data["primary_axis_coords"], qm_data["hessian"]
        )
        if (int(IDs[0]) + 1, int(IDs[1]) + 1) in qm_pairs:
            fc_list.append(
                hessian_analysis.cal_stretching(
                    (int(IDs[0]) + 1, int(IDs[1]) + 1), umatrix, eigmatrix
                )
            )

        elif (int(IDs[1]) + 1, int(IDs[0]) + 1) in qm_pairs:
            fc_list.append(
                hessian_analysis.cal_stretching(
                    (int(IDs[1]) + 1, int(IDs[0]) + 1), umatrix, eigmatrix
                )
            )
        else:
            print(f"Bond not found in QM data: {IDs}")
            continue
    mean_fc = mean(fc_list)

    if output_fc_list:
        return mean_fc, fc_list
    return mean_fc
