import random
import torch
import torch.multiprocessing as mp
from refactor import preprocess, train

if __name__ == "__main__":
    dataset = train.graphDataset(
        "original_hessian_40000", "./graphs/original_hessian_40000_complete_graphs.bin"
    )

    mp.set_sharing_strategy("file_system")
    device = torch.device("cuda")
    seed = random.randrange(0, 10000)
    print("random seed for this training run: ", seed)

    proc = mp.spawn(train.main, args=(1, dataset, seed, 20, 100, "original_hessian_40000", 1000, None, 100, 5, [2, 3]), nprocs=1, join=True)
    # Arguments for train.main:
    # world_size,
    # dataset,
    # seed,
    # total_epoch,
    # patience,
    # save_dataset_name=None,
    # save_freq=0,
    # load_path=None,
    # min_delta=100
    # k = None
    # k_fold_indices = None
