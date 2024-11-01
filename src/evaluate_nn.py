from collections import OrderedDict
import dgl
import pickle
from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader
import pickle
import sys
import torch
from graph_classification import GraphConv, atbDataset
from chemistry_data_structure.helpers.chem import ATOMIC_NUMBER
from chemistry_data_structure.helpers.ir_conversion import wavenumber_to_gromacs_fc


def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value
    return new_state_dict


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

    batched_graph = pickle.load(open("test_batched_graph.pickle", "rb"))
    batched_graph_old = pickle.load(open("test_batched_graph.pickle", "rb")) 
    batched_labels = pickle.load(open("test_batched_labels.pickle", "rb"))
    total_loss = 0
    pred = model(batched_graph, batched_graph.ndata["h"])
    total_loss = torch.abs(pred[:, 0] - batched_labels)

    scaler = pickle.load(open("n_df_scaler.pickle", "rb"))
    unscaled_n = scaler.inverse_transform(batched_graph_old.ndata["h"].numpy())
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
        sum_loss += total_loss[i].item()

    print("Number of ", ATOM_A, " - ", ATOM_B, " pairs: ", len(ids_of_interest))
    print(
        "Average loss for ",
        ATOM_A,
        " - ",
        ATOM_B,
        " is: ",
        wavenumber_to_gromacs_fc(0, ATOM_A, ATOM_B, True, sum_loss / len(ids_of_interest)),
    )
