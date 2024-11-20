from collections import OrderedDict
import dgl
import copy
import pickle
import pickle
import sys
import torch
import matplotlib.pyplot as plt
from graph_classification import GraphConv, atbDataset
from chemistry_data_structure.helpers.chem import ATOMIC_NUMBER
from chemistry_data_structure.helpers.ir_conversion import (
    get_kv_pair_from_FDB,
    wavenumber_to_gromacs_fc,
)
from torch._dynamo.utils import nn_module_new


def draw_wavenumber_hist(wavenumbers, ele_a, ele_b):
    if (ele_a == "C" and ele_b == "H") or (ele_a == "H" and ele_b == "C"):
        plt.axvline(x=3000, color="r", linestyle="--")
        plt.axvline(x=3100, color="r", linestyle="--")
        plt.hist(wavenumbers, bins=100)
    plt.legend(loc="best")
    plt.title(f"{ele_a} - {ele_b}")
    plt.xlabel("Wavenumber ($cm^{-1}$)")
    plt.ylabel("Count")
    plt.show()


def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value
    return new_state_dict


def load_test_data(seed):
    test_set = pickle.load(open(f"test_loader_{seed}.pickle", "rb"))
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
    while True:
        ATOM_A = input("Enter the first atom: ")
        ATOM_B = input("Enter the second atom: ")

        # print('len of total loss is: ', len(total_loss))
        # print('len of batched_graph.edge is ', len(batched_graph.edges()[0]))

        ids_of_interest = []
        idx = 0
        for x, y in zip(batched_graph.edges()[0], batched_graph.edges()[1]):
            if (
                round(unscaled_n[x][0], 3) == ATOMIC_NUMBER[ATOM_A]
                and round(unscaled_n[y][0], 3) == ATOMIC_NUMBER[ATOM_B]
            ) or (
                round(unscaled_n[y][0], 3) == ATOMIC_NUMBER[ATOM_B]
                and round(unscaled_n[x][0], 3) == ATOMIC_NUMBER[ATOM_A]
            ):
                ids_of_interest.append(idx)
            idx += 1

        sum_loss = 0
        for i in ids_of_interest:
            sum_loss += loss_tensor[i].item()

        print("Number of ", ATOM_A, " - ", ATOM_B, " pairs: ", len(ids_of_interest))
        print(
            "Average loss for ",
            ATOM_A,
            " - ",
            ATOM_B,
            " is: ",
            wavenumber_to_gromacs_fc(
                0, ATOM_A, ATOM_B, True, sum_loss / len(ids_of_interest)
            ),
        )


def compare_ir(batched_graph_old, model, scaler):
    test_graph_list = dgl.unbatch(batched_graph_old)
    test_graph_molid = []
    data = atbDataset()
    all_graphs = data.graphs

    for x in all_graphs:
        assert all_graphs[x].ndata["h"].shape[1] == 361

    for tg in test_graph_list:
        for idx, g in enumerate(list(all_graphs.values())):
            if tg.num_nodes() == g.num_nodes() and tg.num_edges() == g.num_edges():
                if torch.equal(tg.ndata["h"], g.ndata["h"]):
                    test_graph_molid.append(idx)
                    break

    for idx, id in enumerate(test_graph_molid):
        test_graph_molid[idx] = list(all_graphs.keys())[id]

    AM_ELEMENT = {value: key for key, value in ATOMIC_NUMBER.items()}
    nn_wavenumbers = []
    fdb_ids = [
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
        "87033"
    ]
    for fdb_id in fdb_ids:
        mol_dict = get_kv_pair_from_FDB(fdb_id)
        for k, v in mol_dict.items():
            if k in test_graph_molid:
                unscaled_k = scaler.inverse_transform(all_graphs[k].ndata["h"])
                for bond in v:
                    for idx, (a, b) in enumerate(
                        zip(all_graphs[k].edges()[0], all_graphs[k].edges()[1])
                    ):
                        if (a, b) == bond:
                            pred = model(
                                copy.deepcopy(all_graphs[k]),
                                copy.deepcopy(all_graphs[k].ndata["h"]),
                            )
                            wavenumber = wavenumber_to_gromacs_fc(
                                0,
                                AM_ELEMENT[round(unscaled_k[a][0], 3)],
                                AM_ELEMENT[round(unscaled_k[b][0], 3)],
                                True,
                                pred[idx].item(),
                            )
                            nn_wavenumbers.append(wavenumber)

    draw_wavenumber_hist(nn_wavenumbers, "C", "H")


if __name__ == "__main__":
    with open("combine_graph_mean.pickle", "rb") as fh:
        myGraph = pickle.load(fh)
    model = GraphConv(dgl.add_self_loop(myGraph).ndata["h"].shape[1], 20, 10)
    checkpoint = torch.load(sys.argv[1], map_location=torch.device("cpu"))
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

    scaler = pickle.load(open("n_df_scaler.pickle", "rb"))
    batched_graph, batched_labels = load_test_data(sys.argv[2])
    batched_graph_old = copy.deepcopy(batched_graph)
    # total_loss = 0
    # the following line modifies batched_graph!!
    # pred = model(batched_graph, batched_graph.ndata["h"])
    # total_loss = torch.abs(pred[:, 0] - batched_labels)

    # compare_qm(total_loss)
    compare_ir(batched_graph_old, model, scaler)
