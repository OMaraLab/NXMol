import os
import re
import statistics
from pprint import pprint
import pickle
import csv
from check import check_everything
from chemistry_data_structure.helpers.ir_conversion import suppress_output
from featurize import (
    load_qm_data,
    merge_edatas,
    printProgressBar,
    write_bonds_edges,
    write_graph,
)
from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D
from numpy import append
from pulp import PulpSolverError
from collections import defaultdict
import matplotlib.pyplot as plt

from chemistry_data_structure.parsing.hessian_analysis import cal_eigen_matrix
from chemistry_data_structure.parsing.hessian_analysis import cal_stretching

from chemistry_data_structure.helpers.graphs import calc_equal_bonds, cull_equal_bonds

# charges = {}
# with open("./netcharges.csv", newline="") as csvfile:
#     data = csv.reader(csvfile, delimiter="\t")
#
#     for row in data:
#         charges[row[0]] = row[1]
#
# net_charge = 0
# bond_tags = []
#
# for idx, x in enumerate(os.listdir("hessian_data")):
#     qm = load_qm_data(x)
#     for id, charge in charges.items():
#         if x == id:
#             net_charge = charge
#     try:
#         mol3D = ATB_QMData_to_Molecule3D(qm, net_charge=int(net_charge), name=x)
#     except (PulpSolverError, Exception):
#         continue
#     for i, j in mol3D.bonds:
#         bond_tags.append(mol3D.BFS_edge(i, j, 1, True))
#     printProgressBar(idx, len(os.listdir("hessian_data")))
#
# pickle.dump(bond_tags, open("bond_tags.pickle", "wb"))

# bond_tags = pickle.load(open("bond_tags.pickle", "rb"))
# fragments = {}
# for idx, (nei_a, nei_b, identifier) in enumerate(bond_tags):
#     ele_a = identifier.split("_")[0][0]
#     ele_b = identifier.split("_")[1][0]
#     if (ele_a, ele_b) not in fragments and (ele_b, ele_a) not in fragments:
#         fragments[(ele_a, ele_b)] = {(nei_a, nei_b): [identifier]}
#     elif (ele_a, ele_b) in fragments:
#         if (nei_a, nei_b) not in fragments[ele_a, ele_b] and (
#             nei_b,
#             nei_a,
#         ) not in fragments[ele_a, ele_b]:
#             fragments[(ele_a, ele_b)][(nei_a, nei_b)] = [identifier]
#         elif (nei_a, nei_b) in fragments[ele_a, ele_b]:
#             fragments[(ele_a, ele_b)][(nei_a, nei_b)].append(identifier)
#         else:
#             fragments[(ele_a, ele_b)][(nei_b, nei_a)].append(identifier)
#     elif (ele_b, ele_a) in fragments:
#         if (nei_a, nei_b) not in fragments[ele_b, ele_a] and (
#             nei_b,
#             nei_a,
#         ) not in fragments[ele_b, ele_a]:
#             fragments[(ele_b, ele_a)][(nei_a, nei_b)] = [identifier]
#         elif (nei_a, nei_b) in fragments[ele_b, ele_a]:
#             fragments[(ele_b, ele_a)][(nei_a, nei_b)].append(identifier)
#         else:
#             fragments[(ele_b, ele_a)][(nei_b, nei_a)].append(identifier)
#     printProgressBar(idx, len(bond_tags))
# pickle.dump(fragments, open("fragments_shell_size_1.pickle", "wb"))
#
#
charges = {}
with open("./netcharges.csv", newline="") as csvfile:
    data = csv.reader(csvfile, delimiter="\t")

    for row in data:
        charges[row[0]] = row[1]

net_charge = 0
fragments = pickle.load(open("fragments_shell_size_1.pickle", "rb"))
# all_fragments = []
# for k_outer, inner_dict in fragments.items():
#     for k_inner, inner_values in inner_dict.items():
#         all_fragments.append((k_outer, k_inner, len(inner_values)))
# sorted_fragments = sorted(all_fragments, key=lambda x: x[2])
# mean_fc = defaultdict(dict)
# for idx, (bond_type, nei, _) in enumerate(sorted_fragments):
#     fc = defaultdict(list)
#     for bond in fragments[bond_type][nei]:
#         fields = re.findall(r"\d+", bond)
#         atoms = re.findall(r"[a-zA-Z]+", bond)
#         qm_data = load_qm_data(fields[2])
#         qm_pairs = [(a, b) for a, b, _ in qm_data["bond_order"]]
#         umatrix, eigmatrix = cal_eigen_matrix(
#             qm_data["primary_axis_coords"], qm_data["hessian"]
#         )
#         if (int(fields[0]) + 1, int(fields[1]) + 1) in qm_pairs:
#             fc[(bond_type, nei)].append(
#                 cal_stretching(
#                     (int(fields[0]) + 1, int(fields[1]) + 1), umatrix, eigmatrix
#                 )
#             )
#         elif (int(fields[1]) + 1, int(fields[0]) + 1) in qm_pairs:
#             fc[(bond_type, nei)].append(
#                 cal_stretching(
#                     (int(fields[1]) + 1, int(fields[0]) + 1), umatrix, eigmatrix
#                 )
#             )
#         else:
#             print(fields, atoms)
#         mean_fc[bond_type][nei] = statistics.mean(fc[bond_type, nei])
#     printProgressBar(idx, len(sorted_fragments))
# pickle.dump(mean_fc, open("mean_fc_shell_size_1.pickle", "wb"))


def load_mol(molID, charges):
    qm_data = load_qm_data(molID)
    net_charge = 0
    for id, charge in charges.items():
        if molID == id:
            net_charge = charge
    silent_mol3D_init = suppress_output(ATB_QMData_to_Molecule3D)
    mol3D = silent_mol3D_init(qm_data, net_charge=int(net_charge), name=molID)
    return mol3D


def update_bond(mol3D, atom1, atom2, mean_fc):
    mol3D.bonds[atom1, atom2].update(force_constant=mean_fc)


def write_full(mol3D, graph_edatas, graph_ndatas, graphs, mean_fc):
    calc_equal_bonds(mol3D)
    cull_equal_bonds(mol3D, "mean", True)
    update_bond(mol3D, atom1, atom2, mean_fc)
    X, y = write_bonds_edges(mol3D, None, True)
    graph_ndatas[mol3D.name], graph_edatas[mol3D.name] = X, y
    graphs[mol3D.name] = write_graph(mol3D)


def write_partial(mol3D, bond, graph_edatas):
    y = write_bonds_edges(mol3D, [bond], False)
    graph_edatas[mol3D.name] = merge_edatas(graph_edatas[mol3D.name], y)


mean_fc = pickle.load(open("mean_fc_shell_size_1.pickle", "rb"))
n_bonds_merged = 0
graph_agg_ndatas = {}
graph_agg_edatas = {}
graphs_agg = {}
mol3D = None
appeared_mols = []
appeared_bonds = []
list_of_nei = []
for bond_type in fragments:
    for idx, nei in enumerate(
        {k: v for k, v in sorted(fragments[bond_type].items(), key=lambda x: len(x[1]))}
    ):
        for bond in fragments[bond_type][nei]:
            atom1, atom2, molID = re.findall(r"\d+", bond)
            # assert (
            #     molID not in appeared_mols or appeared_mols[-1] == molID
            # ), f"appeared_mols[-1] is {appeared_mols[-1]} and molID is {molID}, {appeared_mols}"
            if molID not in graph_agg_edatas:
                mol3D = load_mol(molID, charges)
                write_full(
                    mol3D,
                    graph_agg_edatas,
                    graph_agg_ndatas,
                    graphs_agg,
                    mean_fc[bond_type][nei],
                )
                appeared_mols.append(molID)
            elif molID in graph_agg_edatas and appeared_mols[-1] != molID:
                mol3D = load_mol(molID, charges)
                update_bond(mol3D, atom1, atom2, mean_fc[bond_type][nei])
                write_partial(mol3D, (atom1, atom2), graph_agg_edatas)
                appeared_mols.append(molID)
            elif molID in graph_agg_edatas and appeared_mols[-1] == molID:
                mol3D = load_mol(molID, charges)
                update_bond(mol3D, atom1, atom2, mean_fc[bond_type][nei])
                write_partial(mol3D, (atom1, atom2), graph_agg_edatas)
                appeared_mols.append(molID)
        list_of_nei.append(nei)

        # checking if every bond is aggregated in the current fragmen
        try:
            check_everything(list_of_nei)
        except AssertionError as e:
            print(f"AssertionError: {e}")
            print("list of nei is : ", list_of_nei)
            print("appeared_mols is : ", appeared_mols)
            exit()

        printProgressBar(idx, len(fragments[bond_type]), suffix=f"{bond_type}")


print("n_bonds_merged is ", n_bonds_merged)
pickle.dump(graph_agg_ndatas, open("graph_agg_ndatas_shell_size_1.pickle", "wb"))
pickle.dump(graph_agg_edatas, open("graph_agg_edatas_shell_size_1.pickle", "wb"))
pickle.dump(graphs_agg, open("graphs_agg_shell_size_1.pickle", "wb"))
