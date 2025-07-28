import random
import torch
import torch.multiprocessing as mp
from refactor import preprocess, train

if __name__ == "__main__":
    dataset = train.graphDataset(
        "original_hessian", "./graphs/original_hessian_complete_graphs.bin"
    )

    mp.set_sharing_strategy("file_system")
    device = torch.device("cuda")
    seed = random.randrange(0, 10000)
    print("random seed for this training run: ", seed)

    proc = mp.spawn(train.main, args=(1, dataset, seed, 20, 100), nprocs=1, join=True)
