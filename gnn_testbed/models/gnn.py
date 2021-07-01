import torch
from torch.nn import Linear, ModuleList, Identity, ReLU
import torch.nn.functional as F
from functools import partial
from torch_geometric.nn import (
    TAGConv,
    global_mean_pool,
    global_max_pool,
    BatchNorm,
    global_add_pool,
    TopKPooling,
    Sequential,
    dense_diff_pool,
)


class TAGStackPool(torch.nn.Module):
    def __init__(
        self,
        conv_layer_conf,
        mlp_layer_conf,
        num_node_features,
        num_classes,
        use_batch_norm=False,
        use_skip=False,
    ):
        super(TAGStackPool, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip

        conv_layers = []
        bn_layers = []
        skip_layers = []
        last_out = num_node_features
        for i, (hidden, K, pool) in enumerate(conv_layer_conf):

            seq = []
            if self.use_batch_norm and i > 0:
                seq.append((BatchNorm(last_out), "x -> x"))
            seq += [
                (ReLU(inplace=True), "x -> x"),
                (TAGConv(last_out, hidden, K=K), "x, edge_index -> x"),
            ]

            if pool < 1:
                seq += [
                    (
                        TopKPooling(hidden, ratio=pool),
                        "x, edge_index, edge_attr, batch -> x, edge_index, edge_attr, batch, perm, score",
                    )
                ]

            conv = Sequential("x, edge_index, edge_attr, batch", seq)
            conv_layers.append(conv)

            if self.use_skip:
                if i == 0:
                    skip_layers.append(Identity())
                else:
                    if last_out != hidden:
                        skip_layers.append(
                            Sequential(
                                "x",
                                [
                                    (Linear(2 * last_out, 2 * hidden), "x -> x"),
                                    ReLU(inplace=True),
                                ],
                            )
                        )
                    else:
                        skip_layers.append(Identity())

            last_out = hidden

        self.conv_layers = ModuleList(conv_layers)
        self.skip_layers = ModuleList(skip_layers)

        last_out = 2 * last_out
        mlp = []
        for hidden in mlp_layer_conf:
            mlp.append(Linear(last_out, hidden))
            last_out = hidden
        self.mlp = ModuleList(mlp)
        self.predictor = Linear(last_out, num_classes)

    def forward(self, x, edge_index, batch):

        x_out = 0

        for i, conv in enumerate(self.conv_layers):
            out = conv(x, edge_index, None, batch)
            if isinstance(out, tuple):
                x, edge_index, _, batch, _, _ = out
            else:
                x = out

            if self.use_skip:
                x_skip = torch.cat(
                    [global_mean_pool(x, batch), global_max_pool(x, batch)], axis=1
                )
                x_out = self.skip_layers[i](x_out)
                x_out = x_out + x_skip

        if not self.use_skip:
            x = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], axis=1
            )
        else:
            x = x_out
        x = x.relu()
        for layer in self.mlp:
            x = layer(x).relu()
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.predictor(x)

        return x


class TAGStack(torch.nn.Module):
    def __init__(
        self,
        conv_layer_conf,
        mlp_layer_conf,
        num_node_features,
        num_classes,
        use_batch_norm=False,
        use_skip=False,
    ):
        super(TAGStack, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip

        conv_layers = []
        bn_layers = []
        skip_layers = []
        last_out = num_node_features
        for i, (hidden, K) in enumerate(conv_layer_conf):
            if i == 0:
                conv = TAGConv(last_out, hidden, K=K)
            else:
                if self.use_batch_norm:
                    conv = Sequential(
                        "x, edge_index",
                        [
                            (BatchNorm(last_out), "x -> x"),
                            (ReLU(inplace=True), "x -> x"),
                            (TAGConv(last_out, hidden, K=K), "x, edge_index -> x"),
                        ],
                    )
                else:
                    conv = Sequential(
                        "x, edge_index",
                        [
                            (ReLU(inplace=True), "x -> x"),
                            (TAGConv(last_out, hidden, K=K), "x, edge_index -> x"),
                        ],
                    )

            conv_layers.append(conv)

            if self.use_skip:
                if i == 0:
                    skip_layers.append(Identity())
                else:
                    if last_out != hidden:
                        skip_layers.append(
                            Sequential(
                                "x",
                                [
                                    (Linear(2 * last_out, 2 * hidden), "x -> x"),
                                    ReLU(inplace=True),
                                ],
                            )
                        )
                    else:
                        skip_layers.append(Identity())

            last_out = hidden

        self.conv_layers = ModuleList(conv_layers)
        self.skip_layers = ModuleList(skip_layers)

        last_out = 2 * last_out
        mlp = []
        for hidden in mlp_layer_conf:
            mlp.append(Linear(last_out, hidden))
            last_out = hidden
        self.mlp = ModuleList(mlp)
        self.predictor = Linear(last_out, num_classes)

    def forward(self, x, edge_index, batch):

        x_out = 0
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)

            if self.use_skip:
                x_skip = torch.cat(
                    [global_mean_pool(x, batch), global_max_pool(x, batch)], axis=1
                )
                x_out = self.skip_layers[i](x_out)
                x_out = x_out + x_skip

        if not self.use_skip:
            x = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], axis=1
            )
        else:
            x = x_out
        x = x.relu()
        for layer in self.mlp:
            x = layer(x).relu()
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.predictor(x)

        return x
