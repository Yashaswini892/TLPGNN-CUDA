import argparse
import time
import dgl
import dgl.nn as dglnn
import torch 
import torch.nn as nn
import dgl.function as fn

from dgl import DGLError

import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from dgl.utils import expand_as_pair
from dgl import DGLError


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_size, out_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata["h"] = feat_src

            graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]
            rst = torch.matmul(rst, self.weight)

            rst = rst + self.bias

            return rst


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)



def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    forward = 0
    # training loop
    for epoch in range(10):
        model.train()
        start_time = time.time()  # Record start time
        logits = model(g, features)
        end_time = time.time()  # Record end time
        forward_time = end_time - start_time
          # Calculate forward pass time
        forward += forward_time
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Forward Time {:.4f}".format(
                epoch, loss.item(), acc, forward_time
            )
        )
    print('Time (ms): {:.3f}'.format(forward*1e3/10))
    #print("Total forward time:", forward)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 16, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
