import pickle
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch

"""Some GNN operations are currently *not* deterministic. Check by switching the following to True"""
torch.use_deterministic_algorithms(False)
random.seed(31337)
torch.manual_seed(31337)
torch.cuda.manual_seed(31337)
np.random.seed(31337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(31337)
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
parser.add_argument("--use_bn", dest="use_bn", action="store_true")
parser.add_argument(
    "--model", dest="model", choices=["shallow", "shallow3", "deep"], default="shallow"
)
parser.add_argument(
    "--scheduler",
    dest="scheduler",
    choices=["CosineAnnealingLR", "ReduceLROnPlateau"],
    default="CosineAnnealingLR",
)
args = parser.parse_args()

modules = make_hex_grid(6, 125, 60, 16)
det = Detector(modules)


data_array = torch.load(open("traning_data.pickle", "rb"))

indices = np.arange(len(data_array))
random.shuffle(indices)
shuffled_data = [data_array[i] for i in indices]

split = int(len(shuffled_data) * 0.75)
train_dataset = shuffled_data[:split]
test_dataset = shuffled_data[split:]
test_indices = indices[split:]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(31337)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
)

# 84%


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if args.model == "shallow":
    model = TAGStack(
        [(32, 5), (64, 4), (128, 3), (256, 2), (512, 1)],
        [512, 512],
        num_node_features=15,
        num_classes=5,
        use_batch_norm=args.use_bn,
        use_skip=args.use_skip,
    ).to(device)
elif args.model == "shallow3":
    model = TAGStack(
        [(256, 3), (256, 3), (256, 3), (256, 3), (256, 3)],
        [512, 512],
        num_node_features=15,
        num_classes=5,
        use_batch_norm=args.use_bn,
        use_skip=args.use_skip,
    ).to(device)

elif args.model == "deep":
    model = TAGStack(
        [(32, 5), (64, 4), (128, 3), (128, 3), (128, 3), (256, 2), (512, 1)],
        [512, 512],
        num_node_features=15,
        num_classes=5,
        use_batch_norm=args.use_bn,
        use_skip=args.use_skip,
    ).to(device)


print(count_parameters(model))
writer = SummaryWriter()
model = train_model(
    model,
    train_loader,
    test_loader,
    n_classes=5,
    writer=writer,
    lr=args.lr,
    epochs=args.epochs,
    swa=args.swa,
    swa_lr=args.swa_lr,
    scheduler=args.scheduler,
)
preds, truths, scores = evaluate_model(model, test_loader)

final_acc = (preds == truths).sum() / len(truths)
print("Final accuracy: ", final_acc)
writer.add_hparams(vars(args), {"hparam/accuracy": final_acc})
writer.flush()
writer.close()
