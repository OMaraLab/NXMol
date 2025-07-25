import os
import pickle

import dgl
import dgl.data
import dgl.dataloading
import dgl.function as dfn
import torch
import torch.optim as optim
import torch.nn as tnn
import torch.nn.parallel as tnn_parallel
from torch.distributed import init_process_group, destroy_process_group


class graphDataset(dgl.data.DGLDataset):
    def __init__(self, name, path, url=None):
        self.path = path
        super().__init__(name=name, url=url)

    def process(self):
        graphs, self.molIDs = dgl.load_graphs(self.path)
        assert len(graphs) == len(
            self.molIDs["names"]
        ), "Mismatch in number of graphs and molIDs"
        self._num_graphs = len(graphs)
        self.graphs = None
        del graphs

    def __getitem__(self, idx):
        g, _ = dgl.load_graphs(self.path, [int(idx)])
        return g[0]

    def __len__(self):
        return self._num_graphs


class edgeFeatureSAGEConv(tnn.Module):
    def __init__(
        self,
        in_feats_node,
        in_feats_edge,
        out_feats,
        aggregator_type="mean",
        dropout_rate=0.3,
    ):
        super().__init__()

        self.aggregator_type = aggregator_type

        self.W_msg = tnn.Linear(in_feats_node + in_feats_edge, out_feats)
        self.W_self = tnn.Linear(in_feats_node, out_feats)  # For the self-loop feature
        self.W_concat = tnn.Linear(
            out_feats * 2, out_feats
        )  # For the final concatenation and projection
        self.node_dropout = tnn.Dropout(dropout_rate)

        predictor_input_dim = out_feats * 2

        self.edge_predictor_mlp = tnn.Sequential(
            tnn.Linear(predictor_input_dim, predictor_input_dim // 2),
            tnn.ReLU(),
            tnn.Dropout(dropout_rate),
            tnn.Linear(predictor_input_dim // 2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        tnn.init.xavier_uniform_(self.W_msg.weight)
        tnn.init.zeros_(self.W_msg.bias)
        tnn.init.xavier_uniform_(self.W_self.weight)
        tnn.init.zeros_(self.W_self.bias)
        tnn.init.xavier_uniform_(self.W_concat.weight)
        tnn.init.zeros_(self.W_concat.bias)

        for m in self.edge_predictor_mlp:
            if isinstance(m, tnn.Linear):
                tnn.init.xavier_uniform_(m.weight)
                tnn.init.zeros_(m.bias)

    def forward(self, graph, node_features, edge_features):
        with graph.local_scope():
            graph.ndata["h"] = node_features
            graph.edata["e"] = edge_features

            def message_func(edges):
                combined_features = torch.cat([edges.src["h"], edges.data["e"]], dim=1)
                return {"m": self.W_msg(combined_features)}

            reduce_func = None
            if self.aggregator_type == "mean":
                reduce_func = dfn.mean("m", "h_neigh")
            elif self.aggregator_type == "sum":
                reduce_func = dfn.sum("m", "h_neigh")
            elif self.aggregator_type == "max":
                reduce_func = dfn.max("m", "h_neigh")

            graph.update_all(message_func, reduce_func)

            h_neigh = graph.ndata["h_neigh"]
            h_self = self.W_self(graph.ndata["h"])  # Transform self-node features
            h_combined = torch.cat([h_self, h_neigh], dim=1)

            output_node_features = self.node_dropout(
                tnn.functional.relu(self.W_concat(h_combined))
            )
            graph.ndata["h_out"] = output_node_features

            def edge_score_func(edges):
                combined_edge_input = torch.cat(
                    [edges.src["h_out"], edges.dst["h_out"]], dim=1
                )
                score = self.edge_predictor_mlp(combined_edge_input)
                return {"score": score}

            graph.apply_edges(edge_score_func)

            return graph.edata["score"]


def get_dataloaders(dataset, seed, batch_size):
    train_set, val_set, test_set = dgl.data.split_dataset(
        dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=seed
    )
    train_loader = dgl.dataloading.GraphDataLoader(
        train_set, use_ddp=True, batch_size=batch_size, shuffle=True
    )
    val_loader = dgl.dataloading.GraphDataLoader(val_set, batch_size=batch_size)
    test_loader = dgl.dataloading.GraphDataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def init_model(graph, seed, device, load_path=None):
    epoch_start = 0
    torch.manual_seed(seed)
    model = edgeFeatureSAGEConv(
        graph.ndata["h"].shape[1], graph.edata["e"].shape[1], 64, "mean", 0.3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if load_path:
        map_location = {'cuda:0': str(device)}
        checkpoint = torch.load(load_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = int(checkpoint["epoch"])
    if device.type == "cpu":
        model = tnn_parallel.DistributedDataParallel(model)
    else:
        model = tnn_parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    return model, optimizer, epoch_start


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    for batch in dataloader:
        batched_graph = batch.to(device)
        batched_score = batched_graph.edata["score"].to(device)
        node_feats = batched_graph.ndata["h"].to(device)
        edge_feats = batched_graph.edata["e"].to(device)
        with torch.no_grad():
            predicted_scores = model(batched_graph, node_feats, edge_feats)
            loss = tnn.functional.l1_loss(predicted_scores[:, 0], batched_score)
        total_loss += loss.cpu().item()
        num_batches += 1
    return total_loss / num_batches


def save_model(epoch, model, optimizer, loss, dataset_name):
    os.makedirs("checkpoints", exist_ok=True)
    epoch = str(epoch)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        "checkpoints/{}_epoch_{}.pt".format(dataset_name, epoch),
    )


def main(
    rank,
    world_size,
    dataset,
    seed,
    total_epoch,
    patience,
    save_dataset_name=None,
    save_freq=0,
    load_path=None,
):
    init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = patience
    model, optimizer, epoch_start = init_model(dataset[0], seed, device, load_path)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, seed, batch_size=128
    )
    for epoch in range(total_epoch):
        model.train()
        train_loader.set_epoch(epoch + epoch_start)
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            batched_graph = batch.to(device)
            batched_score = batched_graph.edata["score"].to(device)
            node_feats = batched_graph.ndata["h"].to(device)
            edge_feats = batched_graph.edata["e"].to(device)

            optimizer.zero_grad()
            predicted_scores = model(batched_graph, node_feats, edge_feats)
            loss = tnn.functional.l1_loss(predicted_scores[:, 0], batched_score)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
            num_batches += 1

        if rank == 0:
            print(
                f"Epoch: {epoch_start + epoch + 1}/{epoch_start + total_epoch}, Training loss: {total_loss / num_batches:.4f}"
            )
            if save_freq and (epoch + 1) % save_freq == 0:
                save_model(epoch + 1, model, optimizer, total_loss, save_dataset_name)

        # early stopping

        with torch.no_grad():
            val_loss = torch.tensor(evaluate(model, val_loader, device), device=device)
            if world_size > 1:
                torch.distributed.reduce(
                    val_loss, dst=0, op=torch.distributed.ReduceOp.AVG
                )
        if rank == 0:
            print(
                f"Epoch: {epoch_start + epoch + 1}/{epoch_start + total_epoch}, Validation loss: {val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(
                        f"Early stopping at epoch {epoch_start + epoch + 1}, best validation loss: {best_val_loss:.4f}"
                    )
                    break

    with torch.no_grad():
        train_loss = evaluate(model, train_loader, device)
        val_loss = evaluate(model, val_loader, device)
        test_loss = evaluate(model, test_loader, device)

    print(
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}"
    )
