import pickle
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import Compose, RemoveIsolatedNodes, ToSparseTensor
from torch_geometric.utils import to_networkx
from tqdm.notebook import tqdm

from gnn_testbed.event_generation import (
    make_hex_grid,
    Detector,
)

from gnn_testbed.feature_generation import get_features
from gnn_testbed.event_generation.utils import track_isects_cyl, is_in_cylinder
from gnn_testbed.models import train_model, TAGStack, evaluate_model

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


parser = ArgumentParser()
parser.add_argument("--lr", dest="lr", default=0.01, type=float)
parser.add_argument("--epochs", dest="epochs", default=100, type=int)
parser.add_argument("--swa", dest="swa", action="store_true")
parser.add_argument("--swa_lr", dest="swa_lr", default=0.001, type=float)
parser.add_argument("--use_skip", dest="use_skip", action="store_true")
parser.add_argument(
    "--model", dest="model", choices=["shallow", "deep"], default="shallow"
)
args = parser.parse_args()

modules = make_hex_grid(6, 125, 60, 16)
det = Detector(modules)


data_array = torch.load(open("traning_data.pickle", "rb"))
random.seed(31337)

indices = np.arange(len(data_array))
random.shuffle(indices)
shuffled_data = [data_array[i] for i in indices]

split = int(len(shuffled_data) * 0.75)
train_dataset = shuffled_data[:split]
test_dataset = shuffled_data[split:]
test_indices = indices[split:]

torch.manual_seed(31337)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 84%

if args.model == "shallow":
    model = TAGStack(
        [(32, 5), (64, 4), (128, 3), (256, 2), (512, 1)],
        [512, 512],
        num_node_features=15,
        num_classes=5,
        use_batch_norm=True,
        use_skip=args.use_skip,
    ).to(device)
else:
    model = TAGStack(
        [(32, 5), (64, 4), (128, 3), (128, 3), (128, 3), (256, 2), (512, 1)],
        [512, 512],
        num_node_features=15,
        num_classes=5,
        use_batch_norm=True,
        use_skip=False,
    ).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(count_parameters(model))
writer = SummaryWriter()
model = train_model(
    model,
    train_loader,
    test_loader,
    writer=writer,
    lr=args.lr,
    epochs=args.epochs,
    swa=args.swa,
    swa_lr=args.swa_lr,
)
preds, truths, scores = evaluate_model(model, test_loader)

final_acc = (preds == truths).sum() / len(truths)
print("Final accuracy: ", final_acc)
writer.add_hparams(vars(args), {"hparam/accuracy": final_acc})
writer.flush()
writer.close()
