import sys
sys.path.append('..')

from theano.tensor.nnet import relu

from deepgraph.graph import *
from deepgraph.nn.core import *
from deepgraph.nn.conv import *
from deepgraph.nn.loss import *
from deepgraph.solver import *

from deepgraph.pipeline import Optimizer, H5DBLoader, Pipeline


def build_graph():
    graph = Graph("depth_predictor")

    data            = Data(graph, "data", T.ftensor4, shape=(-1, 3, 288, 384))
    label           = Data(graph, "label", T.ftensor3, shape=(-1, 1, 72, 96), config={
        "phase": PHASE_TRAIN
    })

    conv_0     = Conv2D(graph, "conv_0", config={
            "channels": 96,
            "kernel": (11, 11),
            "subsample": (4, 4),
            "use_cudnn": False,
            "activation": None
        }
    )
    bn = BN(graph, "bn", config={
        "nonlinearity": relu,
        "disable": True
    })
    pool_0 = Pool(graph, "pool_0", config={
        "kernel": (3, 3),
        "use_cudnn": False,
        "stride": (2, 2)
    })
    lrn_0           = LRN(graph, "lrn_0")
    conv_1   = Conv2D(
        graph,
        "conv_1",
        config={
            "channels": 256,
            "kernel": (5, 5),
            "border_mode": 2,
            "activation": relu,
            "use_cudnn": False
        }
    )
    pool_1 = Pool(graph, "pool_1", config={
        "kernel": (3, 3),
        "use_cudnn": False,
        "stride": (2, 2)
    })
    lrn_1           = LRN(graph, "lrn_1")
    conv_2          = Conv2D(
        graph,
        "conv_2",
        config={
            "channels": 384,
            "kernel": (3, 3),
            "use_cudnn": False,
            "border_mode": 1,
            "activation": relu
        }
    )
    conv_3          = Conv2D(
        graph,
        "conv_3",
        config={
            "channels": 384,
            "kernel": (3, 3),
            "border_mode": 1,
            "use_cudnn": False,
            "activation": relu
        }
     )
    conv_4          = Conv2D(
        graph,
        "conv_4",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "use_cudnn": False,
            "border_mode": 1,
            "activation": relu
        }
    )
    pool_4 = Pool(graph, "pool_4", config={
        "kernel": (3, 3),
        "use_cudnn": False,
        "stride": (2, 2)
    })
    flatten         = Flatten(graph, "flatten", config={
        "dims": 2
    })
    hidden_0        = Dense(graph, "fc_0", config={
        "out": 4096,
        "activation": None
    })
    dp_0            = Dropout(graph, "dp_0")
    hidden_1        = Dense(graph, "fc_1", config={
        "out": 6912,
        "activation": None
    })
    rs              = Reshape(graph, "reshape_0", config={
        "shape": (-1, 1, 72, 96),
        "is_output": True
    })

    loss            = EuclideanLoss(graph, "loss")

    error = MSE(graph, "mse", config={
        "root": True,
        "is_output": True,
        "phase": PHASE_TRAIN
    })

    # Connect
    data.connect(conv_0)
    conv_0.connect(bn)
    bn.connect(pool_0)
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

    label.connect(error)
    rs.connect(error)

    return graph


if __name__ == "__main__":

    batch_size = 4
    chunk_size = 10*batch_size
    transfer_shape = ((chunk_size, 3, 288, 384), (chunk_size, 72, 96))

    g = build_graph()

    # Build the training pipeline
    db_loader = H5DBLoader(
        name= "db",
        shapes= ((chunk_size, 3, 480, 640), (chunk_size, 1, 480, 640)),
        config= {
            # "db": '/home/ga29mix/nashome/data/nyu_depth_v2/nyu_depth_v2_sampled.hdf5',
            "db": './data/nyu_depth_v2_sampled.hdf5',
            "key_data": "images",
            "key_label": "depths",
            "chunk_size": chunk_size
    })
    optimizer = Optimizer(
        name= "opt",
        graph= g,
        shapes=transfer_shape,
        config= {
            "batch_size":  batch_size,
            "chunk_size": chunk_size,
            # "learning_rate": 0.01,
            "learning_rate": 0.01,
            "momentum": 0.95,
            "weight_decay": 0.0005,
            "print_freq": 1,
            "save_freq": 30000,
            "weights": "./data/depth_pipeline_alexnet_test_noaug_iter_60000.zip",
            "save_prefix": "./data/depth_pipeline_alexnet_test_noaug"
        }
    )
    p = Pipeline(config={
        "validation_frequency": 50,
        "cycles": 20000
    })
    p.add(db_loader)
    p.add(optimizer)
    p.run()



