import argparse
import time
import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )
        # self.gat_layers.append(
        #     dglnn.GATConv(
        #         hid_size * heads[0],
        #         out_size,
        #         heads[1],
        #         feat_drop=0.6,
        #         attn_drop=0.6,
        #         activation=None,
        #     )
        # )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


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
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    total = 0
    # training loop
    for epoch in range(10):
        model.train()
        start_time = time.time()  # Record start time for the forward pass
        logits = model(g, features)
        end_time = time.time()  # Record end time for the forward pass
        forward_time = end_time - start_time 
        total += forward_time
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        print("Forward Time:", forward_time)
    print('Time (ms): {:.3f}'.format(total*1e3/10))

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
    print(f"Training with DGL built-in GATConv module.")

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

    # create GAT model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GAT(in_size, 8, out_size, heads=[8, 1]).to(device)

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
