import pickle
import json
import os
import csv
import dgl
import random
import numpy as np
import pickle
import statistics

from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D
from pulp import PulpSolverError
from chemistry_data_structure.helpers.graphs import calc_equal_bonds, cull_equal_bonds
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def load_qm_data(molid: str):
    for dirpath, _, filename in os.walk(
        "/home/yaofu/data/atb_fc/NXMol/src/hessian_data"):
        if dirpath == f"/home/yaofu/data/atb_fc/NXMol/src/hessian_data/{molid}":
            with open(f"{dirpath}/{filename[0]}", "rb") as fh:
                return pickle.load(fh)
    raise FileNotFoundError(f"Could not find {molid} in hessian_data")

def load_qm_data_small(molid: str):
    with open(
        f"test_dataset_small/{molid}/b3lyp_631Gd_PCM_water_hessian.pickle", "rb"
    ) as fh:
        return pickle.load(fh)

def write_bonds_edges(mol, bond_list=None, write_X=True):
    X = []
    y = {}
    if bond_list:
        update_list = bond_list
    else:
        update_list = mol.bonds

    for i, j in update_list:
        y[i, j] = mol.bonds[i, j].get("force_constant")

        if write_X == False:
            continue

        if random.random() < 0.5:
            X.append(
                [
                    int(i),
                    mol.atoms[i].element,
                    mol.atoms[i].atomic_number,
                    mol.atoms[i].radius,
                    mol.atoms[i].mass,
                    mol.atoms[i].electronegativity,
                    mol.calcNumBonds(i),
                    int(j),
                    mol.atoms[j].element,
                    mol.atoms[j].atomic_number,
                    mol.atoms[j].radius,
                    mol.atoms[j].mass,
                    mol.atoms[j].electronegativity,
                    mol.calcNumBonds(j),
                    mol.bonds[i, j].get("bond_length"),
                    mol.bonds[i, j].get("fract_bond_order"),
                    mol.BFS_edge(i, j, 1),
                ]
            )
        else:
            X.append(
                [
                    int(j),
                    mol.atoms[j].element,
                    mol.atoms[j].atomic_number,
                    mol.atoms[j].radius,
                    mol.atoms[j].mass,
                    mol.atoms[j].electronegativity,
                    mol.calcNumBonds(j),
                    int(i),
                    mol.atoms[i].element,
                    mol.atoms[i].atomic_number,
                    mol.atoms[i].radius,
                    mol.atoms[i].mass,
                    mol.atoms[i].electronegativity,
                    mol.calcNumBonds(i),
                    mol.bonds[i, j].get("bond_length"),
                    mol.bonds[i, j].get("fract_bond_order"),
                    mol.BFS_edge(j, i, 1),
                ]
            )
    if write_X:
        return X, y
    else:
        return y
                
def write_graph(mol):
    u = []
    v = []
    for i, j in mol.bonds:
        u.append(int(i))
        v.append(int(j))
    tmp = u
    u = u + v
    v = v + tmp
    return dgl.graph((u, v))

def merge_edatas(edata1, edata2):
    for k, v in edata2.items():
        assert bool(k in edata1) != bool(k[::-1] in edata1)
        if k in edata1:
            edata1[k] = v
        elif k[::-1] in edata1:
            edata1[k[::-1]] = v
        assert bool(k in edata1) != bool(k[::-1] in edata1)
    return edata1

if __name__ == "__main__":
    COVALENT_BOND_ORDER_THRESHOLD = 0.5

    from chemistry_data_structure.helpers.ir_conversion import (
        get_distr_from_hessian_FDB, 
        get_molecules_in_FDB_fragment
)
    dirs = os.listdir("hessian_data")
    fdb_ids = os.listdir("fbd")
    charges = {}
    with open("./netcharges.csv", newline="") as csvfile:
        data = csv.reader(csvfile, delimiter="\t")

        for row in data:
            charges[row[0]] = row[1]

    # with open("charges.pickle", "wb") as handle:
    #     pickle.dump(charges, handle)

    graphs_agg = {}
    graph_agg_ndatas = {}
    graph_agg_edatas = {}
    fdb_id = None
    for idx, fdb_id in enumerate(fdb_ids):
        if fdb_id.endswith(".json"):
            fdb_id = fdb_id.split(".")[0]
            bins = {}
            fdb_frag_fcs = get_distr_from_hessian_FDB(fdb_id)[:-1]
            if not fdb_frag_fcs:
                with open("missing_fcs", "a") as fh:
                    fh.write(f"missing fdb_id is {fdb_id}\n")
                continue
            # for fc in fdb_frag_fcs:
            #     bin_num = (fc // 100) * 100
            #     if bin_num not in bins:
            #         bins[bin_num] = []
            #     bins[bin_num].append(fc)
            # max_bin = max(bins, key=lambda x: len(bins[x]))
            # max_bin_mean = statistics.mean(bins[max_bin])
            bin_mean = statistics.mean(fdb_frag_fcs)
            for mol3D, bond_list in get_molecules_in_FDB_fragment(fdb_id, charges):
                for (a, b) in bond_list:
                    a = str(a-1)
                    b = str(b-1)
                    try:
                        mol3D.bonds[a, b].update(force_constant=bin_mean)
                    except KeyError:
                        try:
                            mol3D.bonds[b, a].update(force_constant=bin_mean)
                        except KeyError:
                            with open("missing_bonds", "a") as fh:
                                fh.write(f"fdb_id is {fdb_id}, molID is {mol3D.name}, {a} {b}\n")
                            continue

                assert (mol3D.name not in graph_agg_ndatas 
                        and mol3D.name not in graph_agg_edatas) or \
                        (mol3D.name in graph_agg_ndatas 
                        and mol3D.name in graph_agg_edatas)
                
                X, y = None, None
                if mol3D.name in graph_agg_edatas and mol3D.name in graph_agg_ndatas:
                    try:
                        y = write_bonds_edges(mol3D, [(str(x-1), str(y-1)) for 
                                                    (x, y) in bond_list], False)
                        graph_agg_edatas[mol3D.name] = merge_edatas(
                                                    graph_agg_edatas[mol3D.name], y)
                        # print(mol3D.name, '\n', graph_agg_edatas[mol3D.name].keys(), '\n')
                        for k in graph_agg_edatas[mol3D.name].keys():
                            assert tuple(reversed(k)) not in graph_agg_edatas[mol3D.name].keys()
                    except KeyError:
                        continue
                else:
                    X, y = write_bonds_edges(mol3D)
                    graph_agg_ndatas[mol3D.name] = X
                    graph_agg_edatas[mol3D.name] = y
                    graphs_agg[mol3D.name] = write_graph(mol3D)
            # fdb_mappings = json.load(
            #     open(f"/home/yaofu/data/atb_fc/NXMol/src/fbd/{fdb_id}.json", "r")
            # )["atom_mappings"]
            # for k, v in fdb_mappings.items():
            #     for id in v:
            #         assert (k not in graph_agg_edatas) or \
            #         ((id['1']-1, id['2']-1) not in graph_agg_edatas[k]) or \
            #         ((id['2']-1, id['1']-1) not in graph_agg_edatas[k]) or \
            #         (graph_agg_edatas[k].get(([id['1']-1, id['2']-1])) == max_bin_mean) or \
            #         (graph_agg_edatas[k].get(([id['2']-1, id['1']-1])) == max_bin_mean)
            #
        printProgressBar(idx, len(fdb_ids))

    for x in graph_agg_edatas:
        for y in graph_agg_edatas[x]:
            try:
                assert tuple(reversed(y)) not in graph_agg_edatas[x]
            except AssertionError:
                print(y)
                print(x)
                exit()

    with open("graph_fdb_corrected_ndatas.pickle", "wb") as handle:
        pickle.dump(graph_agg_ndatas, handle)

    with open("graph_fdb_corrected_edatas.pickle", "wb") as handle:
        pickle.dump(graph_agg_edatas, handle)

    with open("graphs_fdb.pickle", "wb") as handle:
        pickle.dump(graphs_agg, handle)
