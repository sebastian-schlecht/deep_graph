import sys
sys.path.append('..')

from deepgraph.utils.logging import log
from deepgraph.utils.common import batch_parallel, ConfigMixin, shuffle_in_unison_inplace, pickle_dump
from deepgraph.utils.image import batch_pad_mirror
from deepgraph.constants import *
from deepgraph.conf import rng

from deepgraph.pipeline import Processor, Packet

from deepgraph.nn.init import *
class Transformer(Processor):
    """
    Apply online random augmentation.
    """
    def __init__(self, name, shapes, config, buffer_size=10):
        super(Transformer, self).__init__(name, shapes, config, buffer_size)
        self.mean = None

    def init(self):
        if self.conf("mean_file") is not None:
            self.mean = np.load(self.conf("mean_file"))
        else:
            log("Transformer - No mean file specified.", LOG_LEVEL_WARNING)

    def process(self):
        packet = self.pull()
        # Return if no data is there
        if not packet:
            return False
        # Unpack
        data, label = packet.data
        # Do processing
        log("Transformer - Processing data", LOG_LEVEL_VERBOSE)

        # Quadratic patches
        w = 400

        start = time.time()
        # Mean
        if packet.phase == PHASE_TRAIN or packet.phase == PHASE_VAL:
            data = data.astype(np.float32)
            if self.mean is not None:
                for idx in range(data.shape[0]):
                    # Subtract mean
                    data[idx] = data[idx] - self.mean.astype(np.float32)
            if self.conf("offset") is not None:
                label -= self.conf("offset")

        if packet.phase == PHASE_TRAIN:
            # Random crops
            cy = rng.randint(data.shape[2] - w, size=1)
            cx = rng.randint(data.shape[3] - w, size=1)
            # cy = (data.shape[2] - w) // 2
            # cx = (data.shape[3] - w) // 2

            data = data[:, :, cy:cy+w, cx:cx+w]
            label = label[:, cy:cy+w, cx:cx+w]

            # Do elementwise operations
            """
            for idx in range(data.shape[0]):
                # Flip with probability 0.5
                p = rng.randint(2)
                if p > 0:
                    data[idx] = data[idx, :, :, ::-1]
                    label[idx] = label[idx, :, ::-1]
                # RGB we mult with a random value between 0.8 and 1.2
                r = rng.randint(80,121) / 100.
                g = rng.randint(80,121) / 100.
                b = rng.randint(80,121) / 100.
                data[idx, 0] = data[idx, 0] * r
                data[idx, 1] = data[idx, 1] * g
                data[idx, 2] = data[idx, 2] * b

            # Shuffle
            data, label = shuffle_in_unison_inplace(data, label)
            """
        elif packet.phase == PHASE_VAL:
            # Center crop
            cy = (data.shape[2] - w) // 2
            cx = (data.shape[3] - w) // 2
            data = data[:, :, cy:cy+w, cx:cx+w]
            label = label[:, cy:cy+w, cx:cx+w]

        end = time.time()
        log("Transformer - Processing took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
        # Try to push into queue as long as thread should not terminate
        self.push(Packet(identifier=packet.id, phase=packet.phase, num=2, data=(data, label)))
        return True

    def setup_defaults(self):
        super(Transformer, self).setup_defaults()
        self.conf_default("mean_file", None)
        self.conf_default("offset", None)


from theano.tensor.nnet import relu

from deepgraph.graph import *
from deepgraph.nn.core import *
from deepgraph.nn.conv import *
from deepgraph.nn.loss import *

from deepgraph.pipeline import Optimizer, H5DBLoader, Pipeline

# Print to console for testing


def build_u_graph():
    graph = Graph("u_depth")

    """
    Inputs
    """
    data = Data(graph, "data", T.ftensor4, shape=(-1, 3, 400, 400))
    label = Data(graph, "label", T.ftensor3, shape=(-1, 1, 400, 400), config={
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
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    conv_2 = Conv2D(
        graph,
        "conv_2",
        config={
            "channels": 64,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
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
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    conv_4 = Conv2D(
        graph,
        "conv_4",
        config={
            "channels": 128,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
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
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    conv_6 = Conv2D(
        graph,
        "conv_6",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
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
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    conv_8 = Conv2D(
        graph,
        "conv_8",
        config={
            "channels": 512,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    pool_8 = Pool(graph, "pool_8", config={
        "kernel": (2, 2)
    })

    """
    Prediction core
    """
    conv_9 = Conv2D(
        graph,
        "conv_9",
        config={
            "channels": 64,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    fl_10 = Flatten(graph, "pc_10", config={
        "dims" : 2
    })
    fc_10 = Dense(graph, "fc_10", config={
            "out": 2500,
            "activation": None,
            "weight_filler": xavier(),
            "bias_filler": constant(0.001)
    })
    rs_10 = Reshape(graph, "rs_10", config={
            "shape": (-1, 1, 50, 50)
    })
    pool_10 = Pool(graph, "pool_10", config={
            "kernel": (2, 2)
    })
    conv_10 = Conv2D(
            graph,
            "conv_10",
            config={
                "channels": 4,
                "kernel": (3, 3),
                "border_mode": 1,
                "activation": relu,
                "weight_filler": xavier(gain="relu"),
                "bias_filler": constant(0)
            }
    )
    """
    Expansive path
    """
    up_11 = Upsample(graph, "up_11", config={
        "kernel": (2, 2)
    })

    conv_12 = Conv2D(
        graph,
        "conv_12",
        config={
            "channels": 512,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    conv_13 = Conv2D(
        graph,
        "conv_13",
        config={
            "channels": 512,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    up_14 = Upsample(graph, "up_14", config={
        "kernel": (2, 2)
    })
    conv_15 = Conv2D(
        graph,
        "conv_15",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "border_mode": 1,
            "weight_filler": xavier(),
            "bias_filler": constant(0)
        }
    )
    conv_16 = Conv2D(
        graph,
        "conv_16",
        config={
            "channels": 256,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )

    up_17 = Upsample(graph, "up_17", config={
        "kernel": (2, 2)
    })
    conv_18 = Conv2D(
        graph,
        "conv_18",
        config={
            "channels": 128,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(),
            "bias_filler": constant(0)
        }
    )
    conv_19 = Conv2D(
        graph,
        "conv_19",
        config={
            "channels": 128,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    up_20 = Upsample(graph, "up_20", config={
        "mode": "constant",
        "kernel": (2, 2)
    })

    conv_21 = Conv2D(
        graph,
        "conv_21",
        config={
            "channels": 64,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(),
            "bias_filler": constant(0)
        }
    )
    conv_22 = Conv2D(
        graph,
        "conv_22",
        config={
            "channels": 64,
            "kernel": (3, 3),
            "border_mode": 1,
            "activation": relu,
            "weight_filler": xavier(gain="relu"),
            "bias_filler": constant(0)
        }
    )
    conv_23 = Conv2D(
        graph,
        "conv_23",
        config={
            "channels": 1,
            "kernel": (1, 1),
            "weight_filler": xavier(),
            "bias_filler": constant(0)
        }
    )

    """
    Feed forward nodes
    """


    concat_20 = Concatenate(graph, "concat_20", config={
        "axis": 1
    })

    concat_17 = Concatenate(graph, "concat_17", config={
        "axis": 1
    })

    concat_14 = Concatenate(graph, "concat_14", config={
        "axis": 1
    })

    concat_11 = Concatenate(graph, "concat_11", config={
        "axis": 1
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
    Drain
    """
    p_drain = Pool(graph, "p_drain", config={
        "kernel": (8,8),
        "mode": "average_inc_pad"
    })

    drain = EuclideanLoss(graph, "drain", config={
        "loss_weight": 0.0
    })


    """
    Make connections
    """
    data.connect(conv_1)
    conv_1.connect(conv_2)
    conv_2.connect(concat_20)
    conv_2.connect(pool_2)

    pool_2.connect(conv_3)
    conv_3.connect(conv_4)
    conv_4.connect(concat_17)
    conv_4.connect(pool_4)
    pool_4.connect(conv_5)
    conv_5.connect(conv_6)
    conv_6.connect(concat_14)
    conv_6.connect(pool_6)
    pool_6.connect(conv_7)
    conv_7.connect(conv_8)
    conv_8.connect(concat_11)
    conv_8.connect(pool_8)
    pool_8.connect(conv_9)
    conv_9.connect(fl_10)
    fl_10.connect(fc_10)
    fc_10.connect(rs_10)
    rs_10.connect(pool_10)
    pool_10.connect(conv_10)
    conv_10.connect(up_11)
    up_11.connect(concat_11)
    concat_11.connect(conv_12)
    conv_12.connect(conv_13)
    conv_13.connect(up_14)
    up_14.connect(concat_14)
    concat_14.connect(conv_15)
    conv_15.connect(conv_16)
    conv_16.connect(up_17)
    up_17.connect(concat_17)
    concat_17.connect(conv_18)
    conv_18.connect(conv_19)
    conv_19.connect(up_20)
    up_20.connect(concat_20)
    concat_20.connect(conv_21)
    conv_21.connect(conv_22)
    conv_22.connect(conv_23)

    conv_23.connect(loss)
    label.connect(loss)

    conv_23.connect(error)
    label.connect(error)

    rs_10.connect(drain)
    label.connect(p_drain)
    p_drain.connect(drain)

    return graph


if __name__ == "__main__":

    batch_size = 8
    chunk_size = 10*batch_size
    transfer_shape = ((chunk_size, 3, 400, 400), (chunk_size, 400, 400))

    g = build_u_graph()

    
    import time
    while True:
	time.sleep(10)
    # Build the training pipeline
    db_loader = H5DBLoader("db", ((chunk_size, 3, 480, 640), (chunk_size, 1, 480, 640)), config={
        "db": '/home/ga29mix/nashome/data/nyu_depth_unet_large.hdf5',
        # "db": '../data/nyu_depth_unet_large.hdf5',
        "key_data": "images",
        "key_label": "depths",
        "chunk_size": chunk_size
    })
    transformer = Transformer("tr", transfer_shape, config={
        # Measured for the data-set
        # "offset": 2.7321029
        "mean_file" : "/home/ga29mix/nashome/data/nyu_depth_unet_large.npy"
    })
    optimizer = Optimizer("opt", g, transfer_shape, config={
        "batch_size":  batch_size,
        "chunk_size": chunk_size,
        "learning_rate": 0.000001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "print_freq": 1,
        "save_freq": 1000,
        # "weights": "../data/vnet_init_2_iter_4500.zip",
        "save_prefix": "../data/vnet_init_3"
    })

    p = Pipeline(config={
        "validation_frequency": 50,
        "cycles": 1000
    })
    p.add(db_loader)
    p.add(transformer)
    p.add(optimizer)
    p.run()

