import numpy as np, h5py
from scipy.misc import imresize

from theano.tensor.nnet import relu

from deepgraph.utils import common
from deepgraph.graph import *
from deepgraph.nn.core import *
from deepgraph.nn.conv import *
from deepgraph.nn.loss import *
from deepgraph.solver import *
from deepgraph.utils.logging import *

from deepgraph.pipeline import Optimizer, H5DBLoader, Pipeline, Transformer


def build_graph():
    graph = Graph("depth_predictor")

    data            = Data(graph, "data", T.ftensor4, shape=(-1, 3, 240, 320))
    label           = Data(graph, "label", T.ftensor3, shape=(-1, 1, 60, 80), phase=PHASE_TRAIN)

    conv_0     = Conv2D(
        graph,
        "conv_0",
        n_channels=96,
        kernel_shape=(11, 11),
        subsample=(4, 4),
        activation=relu
    )
    pool_0 = Pool(graph, "pool_0", kernel_size=(3, 3), stride=(2, 2))
    lrn_0           = LRN(graph, "lrn_0")
    conv_1   = Conv2D(
        graph,
        "conv_1",
        n_channels=256,
        kernel_shape=(5, 5),
        border_mode=2,
        activation=relu
    )
    pool_1 = Pool(graph, "pool_1", kernel_size=(3, 3), stride=(2, 2))
    lrn_1           = LRN(graph, "lrn_1")
    conv_2          = Conv2D(
        graph,
        "conv_2",
        n_channels=384,
        kernel_shape=(3, 3),
        border_mode=1,
        activation=relu
    )
    conv_3          = Conv2D(
        graph,
        "conv_3",
        n_channels=384,
        kernel_shape=(3, 3),
        border_mode=1,
        activation=relu
     )
    conv_4          = Conv2D(
        graph,
        "conv_4",
        n_channels=256,
        kernel_shape=(3, 3),
        border_mode=1,
        activation=relu
    )
    pool_4 = Pool(graph, "pool_4", kernel_size=(3, 3), stride=(2, 2))
    flatten         = Flatten(graph, "flatten", dims=2)
    hidden_0        = FC(graph, "fc_0", n_out=4096, activation=None)
    dp_0            = Dropout(graph, "dp_0")
    hidden_1        = FC(graph, "fc_1", n_out=4800, activation=None)
    rs              = Reshape(graph, "reshape_0", shape=(-1, 1, 60, 80), is_output=True)

    loss            = EuclideanLoss(graph, "loss", loss_weight=1.0)

    # Connect
    data.connect(conv_0)
    conv_0.connect(pool_0)
    pool_0.connect(lrn_0)
    lrn_0.connect(conv_1)
    conv_1.connect(pool_1)
    pool_1.connect(lrn_1)
    lrn_1.connect(conv_2)
    conv_2.connect(conv_3)
    conv_3.connect(conv_4)
    conv_4.connect(pool_4)
    pool_4.connect(flatten)
    flatten.connect(hidden_0)
    hidden_0.connect(dp_0)
    dp_0.connect(hidden_1)
    hidden_1.connect(rs)
    rs.connect(loss)
    label.connect(loss)

    return graph


if __name__ == "__main__":

    batch_size = 64
    chunk_size = 10*batch_size
    transfer_shape = ((chunk_size, 3, 240, 320), (chunk_size, 60, 80))

    g = build_graph()

    # Build the training pipeline
    db_loader = H5DBLoader("db", ((chunk_size, 3, 480, 640), (chunk_size, 1, 480, 640)), config={
        "db": './data/nyu_v2_sampled.hdf5',
        "key_data": "images",
        "key_label": "depths",
        "chunk_size": chunk_size
    })
    transformer = Transformer("tr", transfer_shape, config={})
    optimizer = Optimizer("opt", g, transfer_shape, config={
        "batch_size":  batch_size,
        "chunk_size": chunk_size,
        "iters": 20000,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "print_freq": 1,
        "save_freq": 10000,
        "save_prefix": "./data/depth_pipeline_alexnet"
    })

    p = Pipeline(config={

    })
    p.add(db_loader)
    p.add(transformer)
    p.add(optimizer)
    p.run()



