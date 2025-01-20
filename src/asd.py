import csv
import json
import re
import pickle
from collections import defaultdict
from featurize import load_qm_data
from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D
from chemistry_data_structure.helpers.ir_conversion import get_kv_pair_from_FDB, get_molecules_in_FDB_fragment
from graph_classification import GraphConv, atbDataset, get_dataloaders, init_process_group
from evaluate_nn import remove_module_from_state_dict
import dgl
import torch

if __name__ == "__main__":
    # a = atbDataset()
    # for i in a.graphs:
    #     if a.graphs[i].ndata["h"].shape[1] != 361:
    #         print(i)
    # fragments = pickle.load(open("fragments_shell_size_1.pickle", "rb"))
    # ids = defaultdict(list)
    # for pair in fragments:
    #     for nei in fragments[pair]:
    #         for bond in fragments[pair][nei]:
    #             a1, a2, id = re.findall(r"\d+", bond)
    #             b1, b2 = re.findall(r"[a-zA-Z]+", bond)
    #             print(b1, b2)
    #             ids[id].append((int(a1), int(a2)))

    dataset = atbDataset(raw_dir="graph_original")
    #
    # init_process_group(world_size=1, rank=0)
    # _, _, test_loader = get_dataloaders(dataset, seed=1, batch_size=512)
    # pickle.dump(test_loader, open("graph_original_test_loader_1_512.pickle", "wb"))

    a = pickle.load(open("shell_size_1_files/shell_size_1_test_loader.pickle", "rb"))
    test_graph = dgl.batch([graph for graph in a])
    test_graph_molid = []
    unbatched_list = dgl.unbatch(test_graph)
    for tg in unbatched_list:
        for idx, g in enumerate(list(dataset.graphs.values())):
            if tg.num_nodes() == g.num_nodes() and tg.num_edges() == g.num_edges():
                if torch.equal(tg.ndata["h"], g.ndata["h"]):
                    test_graph_molid.append(idx)
                    break
    with open("combine_graph_mean.pickle", "rb") as fh:
        myGraph = pickle.load(fh)
    model = GraphConv(dgl.add_self_loop(myGraph).ndata["h"].shape[1], 20, 10)
    checkpoint = torch.load('60000_model_state_mean.pt', map_location=torch.device("cpu"))
    checkpoint['model_state_dict'] = remove_module_from_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    list_of_graphs = []
    for y in test_graph_molid:
        list_of_graphs.append(list(dataset.graphs.values())[y])
    final_graph = dgl.batch(list_of_graphs)
    graph_labels = final_graph.edata["score"]
    pred = model(final_graph, final_graph.ndata["h"])
    total_loss = torch.div(torch.abs(pred[:, 0] - graph_labels), graph_labels)
    print(total_loss)
    print(torch.mean(total_loss))
