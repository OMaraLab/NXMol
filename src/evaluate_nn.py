from collections import OrderedDict
import re
from collections import defaultdict
import dgl
import copy
import pickle
import pickle
import sys
from featurize import printProgressBar
import torch
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from graph_classification import GraphConv, atbDataset
from chemistry_data_structure.helpers.chem import ATOMIC_NUMBER
from chemistry_data_structure.helpers.ir_conversion import (
    get_kv_pair_from_FDB,
    wavenumber_to_gromacs_fc,
)
from torch._dynamo.utils import nn_module_new


def check_aggregation(data):
    mol_dict = get_kv_pair_from_FDB("87033")
    remote_escore = pickle.load(open("shell_size_1_e_score.pickle", "rb"))
    id_dict = {}
    for k, v in mol_dict.items():
        if k in data.graphs:
            for x, y in v:
                for idx, (a, b) in enumerate(
                    zip(data.graphs[k].edges()[0], data.graphs[k].edges()[1])
                ):
                    if (a, b) == (x - 1, y - 1) or (b, a) == (x - 1, y - 1):
                        if k in id_dict:
                            id_dict[k].append(idx)
                        else:
                            id_dict[k] = [idx]
    #
    # print(len(id_dict))
    fcs = []
    for k, v in id_dict.items():
        k_id_start = data.e_label.index(k)
        for x in v:
            fcs.append(remote_escore[k_id_start + x])
    print(set(fcs))
    print(len(id_dict.keys()))


def draw_wavenumber_hist(wavenumbers, ele_a, ele_b, save_name=None):
    if (ele_a == "C" and ele_b == "H") or (ele_a == "H" and ele_b == "C"):
        plt.axvline(x=3000, color="r", linestyle="--")
        plt.axvline(x=3100, color="r", linestyle="--")
        plt.hist(wavenumbers, bins=100)
    elif (ele_a == "O" and ele_b == "H") or (ele_a == "H" and ele_b == "O"):
        plt.axvline(x=2700, color="r", linestyle="--")
        plt.axvline(x=3200, color="r", linestyle="--")
        plt.hist(wavenumbers, bins=100)
    elif (ele_a == "C" and ele_b == "N") or (ele_a == "N" and ele_b == "C"):
        plt.axvline(x=2222, color="r", linestyle="--")
        plt.axvline(x=2260, color="r", linestyle="--")
        plt.hist(wavenumbers, bins=100)
    elif (ele_a == "C" and ele_b == "O") or (ele_a == "O" and ele_b == "C"):
        plt.axvline(x=1680, color="r", linestyle="--")
        plt.axvline(x=1750, color="r", linestyle="--")
        plt.hist(wavenumbers, bins=100)

    plt.legend(loc="best")
    plt.title(f"{ele_a} - {ele_b}")
    plt.xlabel("Wavenumber ($cm^{-1}$)")
    plt.ylabel("Count")
    if save_name:
        plt.savefig(f"{ele_a}_{ele_b}_{save_name}.png")
    plt.show()


def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value
    return new_state_dict


def load_test_data(file):
    test_set = pickle.load(open(file, "rb"))
    test_graph = dgl.batch([x for x in test_set])
    test_labels = test_graph.edata["score"]
    return test_graph, test_labels


def compare_qm(loss_tensor):
    scaler = pickle.load(open("n_df_scaler.pickle", "rb"))
    unscaled_n = scaler.inverse_transform(batched_graph_old.ndata["h"])
    AM_ELEMENT = {value: key for key, value in ATOMIC_NUMBER.items()}
    unique_bond_pairs = set()
    for x, y in zip(batched_graph.edges()[0], batched_graph.edges()[1]):
        if (
            AM_ELEMENT[round(unscaled_n[x][0], 3)],
            AM_ELEMENT[round(unscaled_n[y][0], 3)],
        ) in unique_bond_pairs or (
            AM_ELEMENT[round(unscaled_n[y][0], 3)],
            AM_ELEMENT[round(unscaled_n[x][0])],
        ) in unique_bond_pairs:
            continue
        unique_bond_pairs.add(
            (
                AM_ELEMENT[round(unscaled_n[x][0], 3)],
                AM_ELEMENT[round(unscaled_n[y][0], 3)],
            )
        )

    print("Unique bond pairs in the test set: ", unique_bond_pairs)
    print("Overall average loss in gromacs units is: ", torch.mean(loss_tensor).item())
    mean_loss = {}
    for atom1, atom2 in unique_bond_pairs:
        # print('len of total loss is: ', len(total_loss))
        # print('len of batched_graph.edge is ', len(batched_graph.edges()[0]))

        ids_of_interest = []
        idx = 0
        for x, y in zip(batched_graph.edges()[0], batched_graph.edges()[1]):
            if (
                round(unscaled_n[x][0], 3) == ATOMIC_NUMBER[atom1]
                and round(unscaled_n[y][0], 3) == ATOMIC_NUMBER[atom2]
            ) or (
                round(unscaled_n[y][0], 3) == ATOMIC_NUMBER[atom2]
                and round(unscaled_n[x][0], 3) == ATOMIC_NUMBER[atom1]
            ):
                ids_of_interest.append(idx)
            idx += 1

        local_loss_tensor = loss_tensor[ids_of_interest].detach().numpy()
        sum_loss = 0
        for i in ids_of_interest:
            sum_loss += loss_tensor[i].item()
        mean_loss[(atom1, atom2)] = (
            sum_loss / len(ids_of_interest),
            len(ids_of_interest),
        )

        # print("Number of ", ATOM_A, " - ", ATOM_B, " pairs: ", len(ids_of_interest))
        # print(
        #     "Average lossf or ",
        #     ATOM_A,
        #     " - ",
        #     ATOM_B,
        #     " in gromacs units is: ",
        #     sum_loss / len(ids_of_interest),
        # )
        # print(
        #     "Average loss for ",
        #     ATOM_A,
        #     " - ",
        #     ATOM_B,
        #     " in wavenumbers is: ",
        #     wavenumber_to_gromacs_fc(
        #         0, ATOM_A, ATOM_B, True, sum_loss / len(ids_of_interest)
        #     ),
        # )
        #
        # draw_wavenumber_hist(local_loss_tensor, atom1, atom2)
    mean_loss.pop(("H", "H"), None)
    mean_loss.pop(("CL", "S"), None)
    mean_loss = {k: v for k, v in sorted(mean_loss.items(), key=lambda x: x[1])}
    print(mean_loss.values())
    plt.bar(["_".join(x) for x in mean_loss.keys()], [y for y, _ in mean_loss.values()])
    # for i, v in enumerate(mean_loss.values()):
    #     plt.text(i, v, str(v), ha='center')
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    return mean_loss


def compare_ir(batched_graph_old, model, scaler, read_from="fdb"):
    model.eval()
    test_graph_list = dgl.unbatch(batched_graph_old)
    test_graph_molid = []
    data = atbDataset(raw_dir="shell_size_1")
    all_graphs = data.graphs

    for tg in test_graph_list:
        for idx, g in enumerate(list(all_graphs.values())):
            if tg.num_nodes() == g.num_nodes() and tg.num_edges() == g.num_edges():
                if torch.equal(tg.ndata["h"], g.ndata["h"]):
                    test_graph_molid.append(idx)
                    break

    for idx, id in enumerate(test_graph_molid):
        test_graph_molid[idx] = list(all_graphs.keys())[id]

    AM_ELEMENT = {value: key for key, value in ATOMIC_NUMBER.items()}
    ele_a = sys.argv[3]
    ele_b = sys.argv[4]

    ids = None
    nn_wavenumbers = defaultdict(partial(defaultdict, dict))
    if read_from == "fdb":
        if ele_a == "O" and ele_b == "H":
            ids = [
                "87021",
                "87041",
                "87154",
                "87188",
                "87178",
                "87115",
                "87024",
                "87183",
                "87018",
                "87140",
            ]
        elif ele_a == "C" and ele_b == "H":
            ids = [
                "87014",
                "87022",
                "87013",
                "87076",
                "87019",
                "87081",
                "87088",
                "87080",
                "87078",
                "87231",
                "87033",
            ]
        elif ele_a == "C" and ele_b == "N":
            ids = ["87104"]
        elif ele_a == "C" and ele_b == "O":
            ids = ["87304", "87720"]
    elif read_from == "shell_size_1":
        fragments = pickle.load(open("fragments_shell_size_1.pickle", "rb"))
        fragment_lens = pickle.load(
            open("shell_size_1_fragments_by_len.pickle", "rb")
        )

        for idx, x in enumerate(range(len(fragment_lens))):
            ids = defaultdict(list)
            if x < 30:
                pair, nei_pair, _ = list(reversed(fragment_lens))[x]
                for bond in fragments[pair][nei_pair]:
                    a1, a2, id = re.findall(r"\d+", bond)
                    # b1, b2 = re.findall(r"[a-zA-Z]+", bond)
                    # if (b1, b2) == (ele_a, ele_b) or (b2, b1) == (ele_a, ele_b):
                    ids[id].append((int(a1), int(a2)))

                for idxx, id in enumerate(ids):
                    if read_from == "fdb":
                        mol_dict = get_kv_pair_from_FDB(id)
                    elif read_from == "shell_size_1":
                        mol_dict = {id: ids[id]}
                    for k, v in mol_dict.items():
                        if k in test_graph_molid:
                            unscaled_k = scaler.inverse_transform(all_graphs[k].ndata["h"])
                            for bond in v:
                                for idxxx, (a, b) in enumerate(
                                    zip(all_graphs[k].edges()[0], all_graphs[k].edges()[1])
                                ):
                                    if (a, b) == bond or (b, a) == bond:
                                        # print("in here ", k, bond)
                                        with torch.no_grad():
                                            pred = model(
                                                copy.deepcopy(all_graphs[k]),
                                                copy.deepcopy(all_graphs[k].ndata["h"]),
                                            )
                                        wavenumber = wavenumber_to_gromacs_fc(
                                            0,
                                            AM_ELEMENT[round(unscaled_k[a][0], 3)],
                                            AM_ELEMENT[round(unscaled_k[b][0], 3)],
                                            True,
                                            pred[idxxx].item(),
                                        )
                                        if wavenumber == None:
                                            continue
                                        if id not in nn_wavenumbers[pair][nei_pair]:
                                            nn_wavenumbers[pair][nei_pair][id] = [(wavenumber, bond)]
                                        else:
                                            nn_wavenumbers[pair][nei_pair][id].append((wavenumber, bond))
                    printProgressBar(idxx, len(ids))
            printProgressBar(idx, len(fragment_lens))
    pickle.dump(nn_wavenumbers, open("nn_wavenumbers_shell_size_1.pickle", "wb"))
    # draw_wavenumber_hist(nn_wavenumbers, ele_a, ele_b, sys.argv[1])


if __name__ == "__main__":
    with open("combine_graph_mean.pickle", "rb") as fh:
        myGraph = pickle.load(fh)
    model = GraphConv(dgl.add_self_loop(myGraph).ndata["h"].shape[1], 20, 10)
    checkpoint = torch.load(sys.argv[1], map_location=torch.device("cpu"))
    print(checkpoint["model_state_dict"].keys())
    checkpoint["model_state_dict"] = remove_module_from_state_dict(
        checkpoint["model_state_dict"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # dataset = atbDataset()
    # train_set, val_set, test_set = split_dataset(dataset, shuffle=True, random_state=0)
    # test_loader = GraphDataLoader(test_set, batch_size=32)
    # batched_graph_old = dgl.batch([x for x in test_loader])
    # batched_graph = dgl.batch([x for x in test_loader])
    # batched_labels = batched_graph.edata["score"]

    # data = atbDataset()
    scaler = pickle.load(open("n_df_scaler.pickle", "rb"))
    batched_graph, batched_labels = load_test_data(sys.argv[2])
    batched_graph_old = copy.deepcopy(batched_graph)
    total_loss = 0
    # the following line modifies batched_graph!!
    pred = model(batched_graph, batched_graph.ndata["h"])
    # total_loss = torch.abs(pred[:, 0] - batched_labels)
    total_loss = torch.div(torch.abs(pred[:, 0] - batched_labels), batched_labels)
    print(total_loss)

    # a = compare_qm(total_loss)
    # pickle.dump(a, open("loss_dict_mode_agg.pickle", "wb"))
    # check_aggregation(data)
    compare_ir(batched_graph_old, model, scaler, "shell_size_1")
