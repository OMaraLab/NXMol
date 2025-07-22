import torch
import torch.multiprocessing as mp
from refactor import preprocess, train

dataset = preprocess.graphDataset(
    "original_hessian", "/home/yaofu/data/atb_fc/NXMol/src/"
)

mp.set_sharing_strategy("file_descriptor")
device = torch.device("cuda")

proc = mp.spawn(train.main, args=(1, dataset, 0, 10), nprocs=1)
