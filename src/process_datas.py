import pickle
from dgllife.model.model_zoo import WeavePredictor
import dgl
import dgl.nn as nn
import dgl.function as fn
import torch.nn as tnn
import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.data import random_split
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler

def save_obj_2_file(obj, fn):
    with open(fn, 'wb') as fh:
        pickle.dump(obj, fh)

with open('graph_ndatas.pickle', 'rb') as handle:
    ndatas = pickle.load(handle)

with open('graph_edatas.pickle', 'rb') as handle:
    edatas = pickle.load(handle)
    
with open('graphs.pickle', 'rb') as handle:
    graphs = pickle.load(handle)

def OHE_and_normalize(mol: str):
    n = []
    for i in range(graphs[mol].num_nodes()):
        for x in ndatas[mol]:
            if (x[0] == i):
                n.append(x[1:7]+[x[-1][0]]+[mol])
                break
            elif (x[7] == i):
                n.append(x[8:14]+[x[-1][1]]+[mol])
                break
    assert len(n) == graphs[mol].num_nodes()
    
    e = []
    for i, j in zip(graphs[mol].edges()[0].tolist(), graphs[mol].edges()[1].tolist()):
        for x in ndatas[mol]:
            if ((i, j) == (x[0], x[7])) or ((i, j) == (x[7], x[0])):
                e.append(x[-3:-1]+[mol])
                break
        for y in edatas[mol].keys():
            if ((str(i), str(j)) == y) or ((str(j), str(i)) == y):
                e[-1].append([edatas[mol][y], mol, y])
    assert len(e) == graphs[mol].num_edges()
    return n, e

for x in edatas.keys():
    for y in edatas[x]:
        try:
            assert tuple(reversed(y)) not in edatas[x]
        except AssertionError:
            print(y)
            print(x)

sorted_keys = sorted([int(x) for x in ndatas.keys()])
n_all = []
e_all = []
for x in sorted_keys:
        x_n, x_e = OHE_and_normalize(str(x))
        n_all = n_all+x_n
        e_all = e_all+x_e

n_df = pd.DataFrame(n_all, columns=["element", "atomic_number", "radius", "mass", "electronegativity", "hybridisation", "nei", "molID"])
e_df = pd.DataFrame(e_all)
ele_encoded = pd.get_dummies(n_df["element"], prefix="ele_")
nei_encoded = pd.get_dummies(n_df["nei"], prefix="nei")
molID_ndata = n_df["molID"]
n_df.drop("molID", axis=1, inplace=True)
n_df.drop("element", axis=1, inplace=True)
n_df.drop("nei", axis=1, inplace=True)
n_df = n_df.join(ele_encoded)
n_df = n_df.join(nei_encoded)
norm = MinMaxScaler().fit(n_df)
norm_n_df = norm.transform(n_df)

save_obj_2_file(norm, 'shell_size_1_scaler.pickle')
save_obj_2_file(norm_n_df, 'shell_size_1_df.pickle')
save_obj_2_file(molID_ndata, 'shell_size_1_molID_ndata.pickle')

e_star = e_df.drop(2, axis=1)
e_label = e_df[3]
e_star = e_star.drop(3, axis=1)
e_label_star = [i[1] for i in e_label]
norm_e = MinMaxScaler().fit(e_star)
norm_e_star = norm_e.transform(e_star)
e_score = [i[0] for i in e_df[3]]
save_obj_2_file(e_label, "shell_size_1_e_label.pickle")
save_obj_2_file(e_label_star, 'shell_size_1_e_label_star.pickle')
save_obj_2_file(norm_e_star, "shell_size_1_norm_e_star.pickle")
save_obj_2_file(e_score, 'shell_size_1_e_score.pickle')
