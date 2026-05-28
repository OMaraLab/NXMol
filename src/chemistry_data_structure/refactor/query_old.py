import pprint
import time

import numpy as np
import torch

from chemistry_data_structure.helpers.ir_conversion import wavenumber_to_gromacs_fc
from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D, pdb_to_Molecule3D
from chemistry_data_structure.refactor.featurize import load_qm_data
from chemistry_data_structure.refactor.train import (
    edgeFeatureSAGEConv,
    graphDataset,
    init_model,
    init_process_group,
)
from chemistry_data_structure.refactor.utils import load_charges

# step 1: load charges and molecule data; you need them to know which atom type to pass to wavenumber_to_gromacs_fc
charges = load_charges("./netcharges_40000.csv")
qm_data = load_qm_data("21", "./hessian_data_40000")
mol3D = ATB_QMData_to_Molecule3D(qm_data, charges["21"])
# pdb_mol = pdb_to_Molecule3D(open("./_I0L_allatom_optimised_geometry.pdb", 'r').read())


# step 2, load dataset and model
dataset = graphDataset(
    "original_hessian_40000", "./graphs/original_hessian_40000_complete_graphs.bin"
)

device = torch.device("cpu")
init_process_group("gloo", init_method="tcp://127.0.0.1:12345", world_size=1, rank=0)
checkpoins = ["./checkpoints/original_hessian_40000_best_fold_6_of_10.pt"]

# step 3, inference
# start_time = time.perf_counter()
graph = dataset[29518]
# graph = dataset[14294]
tol_freq = np.zeros((len(checkpoins), graph.num_edges()))

# start_time = time.process_time()
for idy, point in enumerate(checkpoins):
    model, _, _ = init_model(graph, 0, device, point)
    model.eval()
    with torch.no_grad():
        tol_fc = model(graph, graph.ndata["h"], graph.edata["e"])
        for idx, (x, y) in enumerate(
            zip(tol_fc, zip(graph.edges()[0], graph.edges()[1]))
        ):
            freq = wavenumber_to_gromacs_fc(
                None,
                mol3D.atoms[str(y[0].item())].element,
                mol3D.atoms[str(y[1].item())].element,
                backward=True,
                gmx_fc=x.item(),
            )
            # print(idx)
            tol_freq[idy][idx] += freq
end_time = time.process_time()
# print(f"Time taken: {end_time - start_time} seconds")

std = np.std(tol_freq, axis=0).tolist()

for x, y in zip(std, zip(dataset[29518].edges()[0], dataset[29518].edges()[1])):
    print(y, x)


seminario_freq = []
for x in mol3D.bonds:
    seminario_freq.append(
        wavenumber_to_gromacs_fc(
            None,
            mol3D.atoms[x[0]].element,
            mol3D.atoms[x[1]].element,
            backward=True,
            gmx_fc=mol3D.bonds[x].get("force_constant"),
        )
    )

pred_freq = []
for x in mol3D.bonds:
    pred_freq.append(
        wavenumber_to_gromacs_fc(
            None,
            mol3D.atoms[x[0]].element,
            mol3D.atoms[x[1]].element,
            backward=True,
            gmx_fc=mol3D.bonds[x].get("force_constant"),
        )
    )

pprint.pprint(tol_freq)
mol3D.draw_graph(node_label_mode="id", node_size=15)
