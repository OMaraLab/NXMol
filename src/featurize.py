import pickle
import os
import csv
import dgl
import random
import numpy as np
import pickle

from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D


def load_qm_data(molid: str):
    for dirpath, dirname, filename in os.walk("test_dataset_big"):
        if dirpath == f"test_dataset_big/{molid}":
            with open(f"{dirpath}/{filename[0]}", "rb") as fh:
                return pickle.load(fh)


def load_qm_data_small(molid: str):
    with open(
        f"test_dataset_small/{molid}/b3lyp_631Gd_PCM_water_hessian.pickle", "rb"
    ) as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    COVALENT_BOND_ORDER_THRESHOLD = 0.5

    dirs = os.listdir("test_dataset_big")
    charges = []
    with open("./netcharges.csv", newline="") as csvfile:
        data = csv.reader(csvfile, delimiter="\t")

        for idx, row in enumerate(data):
            charges.append((row[0], row[1]))

    graphs = {}
    graph_ndatas = {}
    graph_edatas = {}
    net_charge = 0
    idx = 0
    for mol in dirs:
        X = []
        y = {}
        for id, charge in charges:
            if mol == id:
                net_charge = charge
        qm_data = load_qm_data(mol)
        mol3D = ATB_QMData_to_Molecule3D(qm_data, net_charge=int(net_charge), name=mol)
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
            idx += 1
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

    with open("graph_ndatas.pickle", "wb") as handle:
        pickle.dump(graph_ndatas, handle)

    with open("graph_edatas.pickle", "wb") as handle:
        pickle.dump(graph_edatas, handle)

    with open("graphs.pickle", "wb") as handle:
        pickle.dump(graphs, handle)
