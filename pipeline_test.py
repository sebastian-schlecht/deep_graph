from theano.tensor.nnet import relu

from deepgraph.graph import *
from deepgraph.nn.core import *
from deepgraph.nn.conv import *
from deepgraph.nn.loss import *
from deepgraph.solver import *

from deepgraph.pipeline import Optimizer, H5DBLoader, Pipeline, Transformer


def build_graph():
    graph = Graph("depth_predictor")

    data            = Data(graph, "data", T.ftensor4, shape=(-1, 3, 240, 320))
    label           = Data(graph, "label", T.ftensor3, shape=(-1, 1, 60, 80), config={
        "phase": PHASE_TRAIN
    })

    conv_0     = Conv2D(graph, "conv_0", config={
            "channels": 96,
            "kernel": (11, 11),
            "subsample": (4, 4),
            "activation": relu
        }
    )
    pool_0 = Pool(graph, "pool_0", config={
        "kernel": (3, 3),
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
            "activation": relu
        }
    )
    pool_1 = Pool(graph, "pool_1", config={
        "kernel": (3, 3),
        "stride": (2, 2)
    })
    lrn_1           = LRN(graph, "lrn_1")
    conv_2          = Conv2D(
        graph,
        "conv_2",
        config={
            "channels": 384,
            "kernel": (3, 3),
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
            "activation": relu
        }
     )
    conv_4          = Conv2D(
        graph,
        "conv_4",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu
        }
    )
    pool_4 = Pool(graph, "pool_4", config={
        "kernel": (3, 3),
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
        "out": 4800,
        "activation": None
    })
    rs              = Reshape(graph, "reshape_0", config={
        "shape": (-1, 1, 60, 80),
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

    label.connect(error)
    rs.connect(error)

    return graph


def build_u_graph():
    graph = Graph("u_depth")

    """
    Inputs
    """
    data = Data(graph, "data", T.ftensor4, shape=(-1, 3, 240, 320))
    label = Data(graph, "label", T.ftensor3, shape=(-1, 1, 60, 80), config={
        "phase": PHASE_TRAIN
    })
    """
    Contractive part
    """
    conv_1 = Conv2D(
        graph,
        "conv_1",
        config={
            "channels": 64,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_2 = Conv2D(
        graph,
        "conv_2",
        config={
            "channels": 64,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    pool_2 = Pool(graph, "pool_2", config={
        "kernel": (2, 2)
    })
    conv_3 = Conv2D(
        graph,
        "conv_3",
        config={
            "channels": 128,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_4 = Conv2D(
        graph,
        "conv_4",
        config={
            "channels": 128,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    pool_4 = Pool(graph, "pool_4", config={
        "kernel": (2, 2)
    })

    conv_5 = Conv2D(
        graph,
        "conv_5",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_6 = Conv2D(
        graph,
        "conv_6",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )

    pool_6 = Pool(graph, "pool_6", config={
        "kernel": (2, 2)
    })

    conv_7 = Conv2D(
        graph,
        "conv_7",
        config={
            "channels": 512,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_8 = Conv2D(
        graph,
        "conv_8",
        config={
            "channels": 512,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    pool_8 = Pool(graph, "pool_8", config={
        "kernel": (2, 2)
    })

    conv_9 = Conv2D(
        graph,
        "conv_9",
        config={
            "channels": 1024,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_10 = Conv2D(
            graph,
            "conv_10",
            config={
                "channels": 1024,
                "kernel": (3, 3),
                "subsample": (1, 1),
                "activation": relu
            }
    )

    """
    Expansive path
    """
    up_11 = Upsample(graph, "up_11", config={
        "kernel": (2, 2)
    })
    conv_11 = Conv2D(
        graph,
        "conv_11",
        config={
            "channels": 512,
            "kernel": (2, 2),
            "activation": relu
        }
    )
    # TODO Insert concat
    conv_12 = Conv2D(
        graph,
        "conv_12",
        config={
            "channels": 512,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_13 = Conv2D(
        graph,
        "conv_13",
        config={
            "channels": 512,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    up_14 = Upsample(graph, "up_14", config={
        "kernel": (2, 2)
    })
    conv_14 = Conv2D(
        graph,
        "conv_14",
        config={
            "channels": 256,
            "kernel": (2, 2),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    # TODO Insert concat
    conv_15 = Conv2D(
        graph,
        "conv_15",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_16 = Conv2D(
        graph,
        "conv_16",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )

    up_17 = Upsample(graph, "up_17", config={
        "kernel": (2, 2)
    })
    conv_17 = Conv2D(graph, "conv_17", config={
            "channels": 128,
            "kernel": (2, 2),
            "subsample": (1, 1),
            "activation": relu
    })
    conv_18 = Conv2D(
        graph,
        "conv_18",
        config={
            "channels": 128,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_19 = Conv2D(
        graph,
        "conv_19",
        config={
            "channels": 128,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    up_20 = Upsample(graph, "up_20", config={
        "kernel": (2, 2)
    })
    conv_20 = Conv2D(graph, "conv_20", config={
            "channels": 64,
            "kernel": (2, 2),
            "subsample": (1, 1),
            "activation": relu
    })
    conv_21 = Conv2D(
        graph,
        "conv_21",
        config={
            "channels": 64,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_22 = Conv2D(
        graph,
        "conv_22",
        config={
            "channels": 64,
            "kernel": (3, 3),
            "subsample": (1, 1),
            "activation": relu
        }
    )
    conv_23 = Conv2D(
        graph,
        "conv_23",
        config={
            "channels": 1,
            "kernel": (1, 1),
            "activation": relu
        }
    )

    fl = Flatten(graph, "flatten", config={
        "dims": 2
    })

    fc_1 = Dense(graph, "fc_1", config={
        "out": 4800,
        "activation": None
    })
    rs = Reshape(graph, "reshape_0", config={
        "shape": (-1, 1, 60, 80),
        "is_output": True
    })


    """
    Losses / Error
    """
    loss = EuclideanLoss(graph, "loss")

    error = MSE(graph, "mse", config={
        "root": True,
        "is_output": True,
        "phase": PHASE_TRAIN
    })


    """
    Make connections
    """
    data.connect(conv_1)
    conv_1.connect(conv_2)
    conv_2.connect(pool_2)
    pool_2.connect(conv_3)
    conv_3.connect(conv_4)
    conv_4.connect(pool_4)
    pool_4.connect(conv_5)
    conv_5.connect(conv_6)
    conv_6.connect(pool_6)
    pool_6.connect(conv_7)
    conv_7.connect(conv_8)
    conv_8.connect(pool_8)
    pool_8.connect(conv_9)
    conv_9.connect(conv_10)
    conv_10.connect(up_11)
    up_11.connect(conv_11)
    conv_11.connect(conv_12)
    conv_12.connect(conv_13)
    conv_13.connect(up_14)
    up_14.connect(conv_14)
    conv_14.connect(conv_15)
    conv_15.connect(conv_16)
    conv_16.connect(up_17)
    up_17.connect(conv_17)
    conv_17.connect(conv_18)
    conv_18.connect(conv_19)
    conv_19.connect(up_20)
    up_20.connect(conv_20)
    conv_20.connect(conv_21)
    conv_21.connect(conv_22)
    conv_22.connect(conv_23)

    conv_23.connect(fl)
    fl.connect(fc_1)
    fc_1.connect(rs)
    rs.connect(loss)

    label.connect(loss)

    rs.connect(error)
    label.connect(error)

    return graph



if __name__ == "__main__":

    batch_size = 64
    chunk_size = 10*batch_size
    transfer_shape = ((chunk_size, 3, 240, 320), (chunk_size, 60, 80))

    g = build_u_graph()

    # Build the training pipeline
    db_loader = H5DBLoader("db", ((chunk_size, 3, 480, 640), (chunk_size, 1, 480, 640)), config={
        # "db": '/home/ga29mix/nashome/data/nyu_depth_v2/nyu_depth_v2_sampled.hdf5',
        "db": 'data/nyu_depth_combined.hdf5',
        "key_data": "images",
        "key_label": "depths",
        "chunk_size": chunk_size
    })
    transformer = Transformer("tr", transfer_shape, config={
        # Measured empirically for the data-set
        "offset": 2.7321029
    })
    optimizer = Optimizer("opt", g, transfer_shape, config={
        "batch_size":  batch_size,
        "chunk_size": chunk_size,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "print_freq": 1,
        "save_freq": 30000,
        "weights": "data/depth_pipeline_alexnet_test_noaug_iter_60000.zip",
        "save_prefix": "./data/depth_pipeline_alexnet_test_noaug"
    })

    p = Pipeline(config={
        "validation_frequency": 1,
        "cycles": 20000
    })
    p.add(db_loader)
    p.add(transformer)
    p.add(optimizer)
    p.run()



