import pickle
import os
import csv
import dgl
import random
import numpy as np
import pickle

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
    for dirpath, dirname, filename in os.walk("hessian_data"):
        if dirpath == f"hessian_data/{molid}":
            with open(f"{dirpath}/{filename[0]}", "rb") as fh:
                return pickle.load(fh)


def load_qm_data_small(molid: str):
    with open(
        f"test_dataset_small/{molid}/b3lyp_631Gd_PCM_water_hessian.pickle", "rb"
    ) as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    COVALENT_BOND_ORDER_THRESHOLD = 0.5

    dirs = os.listdir("hessian_data")
    charges = []
    with open("./netcharges.csv", newline="") as csvfile:
        data = csv.reader(csvfile, delimiter="\t")

        for idx, row in enumerate(data):
            charges.append((row[0], row[1]))

    print()
    with open("charges.pickle", "wb") as handle:
        pickle.dump(charges, handle)

    graphs = {}
    graph_ndatas = {}
    graph_edatas = {}
    net_charge = 0
    for idx, mol in enumerate(dirs):
        X = []
        y = {}
        for id, charge in charges:
            if mol == id:
                net_charge = charge
        qm_data = load_qm_data(mol)
        try:
            mol3D = ATB_QMData_to_Molecule3D(qm_data, net_charge=int(net_charge), name=mol)
        except (PulpSolverError, Exception):
            continue
        calc_equal_bonds(mol3D)
        cull_equal_bonds(mol3D, graph=True)
        for i, j in mol3D.bonds:
            y[i, j] = mol3D.bonds[i, j].get("force_constant")

            if random.random() < 0.5:
                X.append(
                    [
                        int(i),
                        mol3D.atoms[i].element,
                        mol3D.atoms[i].atomic_number,
                        mol3D.atoms[i].radius,
                        mol3D.atoms[i].mass,
                        mol3D.atoms[i].electronegativity,
                        mol3D.calcNumBonds(i),
                        int(j),
                        mol3D.atoms[j].element,
                        mol3D.atoms[j].atomic_number,
                        mol3D.atoms[j].radius,
                        mol3D.atoms[j].mass,
                        mol3D.atoms[j].electronegativity,
                        mol3D.calcNumBonds(j),
                        mol3D.bonds[i, j].get("bond_length"),
                        mol3D.bonds[i, j].get("fract_bond_order"),
                        mol3D.BFS_edge(i, j, 1),
                    ]
                )
            else:
                X.append(
                    [
                        int(j),
                        mol3D.atoms[j].element,
                        mol3D.atoms[j].atomic_number,
                        mol3D.atoms[j].radius,
                        mol3D.atoms[j].mass,
                        mol3D.atoms[j].electronegativity,
                        mol3D.calcNumBonds(j),
                        int(i),
                        mol3D.atoms[i].element,
                        mol3D.atoms[i].atomic_number,
                        mol3D.atoms[i].radius,
                        mol3D.atoms[i].mass,
                        mol3D.atoms[i].electronegativity,
                        mol3D.calcNumBonds(i),
                        mol3D.bonds[i, j].get("bond_length"),
                        mol3D.bonds[i, j].get("fract_bond_order"),
                        mol3D.BFS_edge(j, i, 1),
                    ]
                )
        graph_ndatas[mol] = X
        graph_edatas[mol] = y
        u = []
        v = []
        for i, j in mol3D.bonds:
            u.append(int(i))
            v.append(int(j))
        tmp = u
        u = u + v
        v = v + tmp
        graphs[mol] = dgl.graph((u, v))
        printProgressBar(idx, len(dirs))

    with open("graph_mean_ndatas.pickle", "wb") as handle:
        pickle.dump(graph_ndatas, handle)

    with open("graph_mean_edatas.pickle", "wb") as handle:
        pickle.dump(graph_edatas, handle)

    with open("graphs_mean.pickle", "wb") as handle:
        pickle.dump(graphs, handle)
