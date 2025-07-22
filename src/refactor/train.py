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


class EdgeFeatureSAGEConv(tnn.Module):
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
        self.edge_predictor = tnn.Linear(predictor_input_dim, 1)

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
                tnn.ReLU(self.W_concat(h_combined))
            )
            graph.ndata["h_out"] = output_node_features

            def edge_score_func(edges):
                combined_edge_input = torch.cat(
                    [edges.src["h_out"], edges.dst["h_out"]], dim=1
                )
                score = self.edge_predictor(combined_edge_input)
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


def init_model(dataset, seed, device):
    torch.manual_seed(seed)
    model = EdgeFeatureSAGEConv(
        dataset.ndata["h"].shape[1], dataset.edata["e"].shape[1], 64, "mean", 0.3
    )
    if device.type == "cpu":
        model = tnn_parallel.DistributedDataParallel(model)
    else:
        model = tnn_parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    return model


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
    epoch = str(epoch)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        "./checkpoints/{}_epoch_{}.pt".format(dataset_name, epoch),
    )


def main(
    rank,
    world_size,
    dataset,
    seed,
    total_epoch,
    load=None,
    save_dataset_name=None,
    save_freq=0,
):
    init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model = init_model(dataset, seed, device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch_start = 0
    if load:
        checkpoint = torch.load(load, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, seed, batch_size=32
    )
    for epoch in range(total_epoch):
        model.train()
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

        print(
            f"Epoch: {epoch_start + epoch + 1}/{total_epoch}, Loss: {total_loss / num_batches:.4f}"
        )
        if save_freq and (epoch + 1) % save_freq == 0:
            save_model(epoch + 1, model, optimizer, total_loss, save_dataset_name)

    with torch.no_grad():
        train_loss = evaluate(model, train_loader, device)
        val_loss = evaluate(model, val_loader, device)
        test_loss = evaluate(model, test_loader, device)

    print(
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}"
    )

    destroy_process_group()
