import numpy as np, h5py
from scipy.misc import imresize
from scipy.ndimage import zoom

import theano
from theano.tensor.nnet import relu


from deepgraph.utils import common
from deepgraph.graph import *
from deepgraph.nn.core import *
from deepgraph.nn.conv import *
from deepgraph.nn.loss import *
from deepgraph.solver import *
from deepgraph.utils.logging import *


def load_data(db_file):
    # Read the MAT-File into memory
    log("Loading data from database", LOG_LEVEL_INFO)
    dataset = h5py.File(db_file)

    depth_field = dataset['depths']
    depths = np.array(depth_field)

    images_field = dataset['images']
    images = np.array(images_field).astype(np.uint8)

    # Swap axes
    images = np.swapaxes(images, 2, 3)
    depths = np.swapaxes(depths, 1, 2)

    # Resizing
    log("Resizing input", LOG_LEVEL_INFO)
    img_scale = 0.5
    depth_scale = 0.125

    images_sized = np.zeros((images.shape[0], images.shape[1], int(images.shape[2]*img_scale), int(images.shape[3]*img_scale)), dtype=np.uint8)
    depths_sized = np.zeros((depths.shape[0], int(depths.shape[1]*depth_scale), int(depths.shape[2]*depth_scale)), dtype=np.float32)

    for i in range(len(images)):
        ii = imresize(images[i], img_scale)
        images_sized[i] = np.swapaxes(np.swapaxes(ii, 1, 2), 0, 1)

    # For this test, we down-sample the depth images to 64x48

    for d in range(len(depths)):
        dd = zoom(depths[d], depth_scale)
        depths_sized[d] = dd

    images = images_sized
    depths = depths_sized

    images_train, images_val = common.split_array(images, 0.9)
    depths_train, depths_val = common.split_array(depths, 0.9)
    return [(images_train, images_val),(depths_train, depths_val)]


def build_graph():
    graph = Graph("depth_predictor")
    data_in = Data(graph, "data", T.ftensor4, shape=(-1, 3, 240, 320))
    label = Data(graph, "label", T.ftensor3, shape=(-1, 1, 60, 80), config={
        "phase": PHASE_TRAIN
    })

    conv_1 = Conv2D(graph, "conv_1", config={
        "n_channels": 96,
        "kernel_shape": (11, 11),
        "subsample": (4, 4),     # Stride
        "activation": relu,
        "weight_filler": normal(),
        "bias_filler": constant(0)
    })
    lrn_1 = LRN(graph, "lrn_1")

    pool_1 = Pool(graph, "pool_1", config={
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "ignore_border": True
    })

    conv_2 = Conv2D(graph, "conv_2", config={
        "n_channels": 256,
        "border_mode": (2, 2),
        "kernel_shape": (5, 5),
        "weight_filler": normal(),
        "bias_filler": constant(1),
        "activation": relu
    })

    pool_2 = Pool(graph, "pool_2", config={
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "ignore_border": True
    })

    lrn_2 = LRN(graph, "lrn_2")

    conv_3 = Conv2D(graph, "conv_3", config={
        "n_channels": 384,
        "border_mode": (1, 1),
        "kernel_shape": (3, 3),
        "weight_filler": normal(),
        "bias_filler": constant(0),
        "activation": relu
    })
    conv_4 = Conv2D(graph, "conv_4", config={
        "n_channels": 384,
        "border_mode": (1, 1),
        "kernel_shape": (3, 3),
        "weight_filler": normal(),
        "bias_filler": constant(1),
        "activation": relu
    })
    conv_5 = Conv2D(graph, "conv_5", config={
        "n_channels": 256,
        "border_mode": (1, 1),
        "kernel_shape": (3, 3),
        "weight_filler": normal(),
        "bias_filler": constant(1),
        "activation": relu
    })

    pool_5 = Pool(graph, "pool_5", config={
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "ignore_border": True
    })

    flatten_5 = Flatten(graph, "flatten_5", config={
        "dims": 2
    })

    fc6 = Dense(graph, "fc6", config={
        "n_out": 4096,
        "activation": None,
        "weight_filler": normal(0, 0.005),
        "bias_filler": constant(1)
    })

    #dp6 = Dropout(graph, "dp6")

    fc7 = Dense(graph, "fc7", config={
        "n_out": 4096,
        "activation": None,
        "weight_filler": normal(0, 0.005),
        "bias_filler": constant(1)
    })

    # dp7 = Dropout(graph, "dp7")

    fc8 = Dense(graph, "fc8", config={
        "n_out": 4800,
        "activation": relu,
        "weight_filler": normal(),
        "bias_filler": constant(0.1),

    })

    rs9 = Reshape(graph, "rs9", config={
        "shape": (-1, 1, 60, 80),
        "is_output": True
    })

    loss            = EuclideanLoss(graph, "loss")
    l1 = L2RegularizationLoss(graph, "l1", loss_weight=0.001)

    # Connect
    data.connect(conv_pool_0)
    conv_pool_0.connect(lrn_0)
    lrn_0.connect(conv_pool_1)
    conv_pool_1.connect(lrn_1)
    lrn_1.connect(conv_2)
    conv_2.connect(conv_3)
    conv_3.connect(conv_4)
    conv_4.connect(flatten)
    flatten.connect(hidden_0)
    hidden_0.connect(hidden_1)
    hidden_1.connect(hidden_2)
    hidden_2.connect(rs)
    rs.connect(loss)
    label.connect(loss)

    hidden_0.connect(l1)
    hidden_1.connect(l1)

    return graph


if __name__ == "__main__":
    data = load_data('/home/ga29mix/nashome/data/nyu_depth_v2/nyu_depth_v2_labeled.mat')
    # data = load_data('./data/nyu_depth_v2_labeled.mat')
    train_x, val_x = data[0]
    train_y, val_y = data[1]

    #######################
    # Data preprocessing
    #######################
    log("Preprocessing data", LOG_LEVEL_INFO)

    # X
    # Scale into 0-1 range
    # train_x = train_x.astype(np.float)
    # train_x *= 0.003921
    # Subtract mean
    train_x = train_x.astype(np.float32)
    train_mean = np.mean(train_x, axis=0)
    train_mean = train_mean.astype(np.float32)
    np.save("train_mean.npy", train_mean)
    for i in range(train_x.shape[0]):
        train_x[i] = train_x[i] - train_mean
    # Y
    # Scale down by 100
    #train_y *= 0.01

    # Wrap data into theano shared variables
    var_train_x = common.wrap_shared(train_x.astype(np.float32))
    var_train_y = common.wrap_shared(train_y)

    var_val_x = common.wrap_shared(val_x.astype(np.float32))
    var_val_y = common.wrap_shared(val_y)

    batch_size = 64

    g = build_graph()
    model_file = "data/model.zip"
    # g.load_weights(model_file)
    base_lr = 0.001
    solver = Solver(lr=base_lr)
    solver.compile_and_fit(
        graph=g,
        epochs=10,
        train_input=(train_x.astype(np.float32), train_y),
        batch_size=batch_size,
        superbatch_size=2*batch_size,
        print_freq=1
    )
    log("Saving final model", LOG_LEVEL_INFO)
    g.save(model_file)
    """
    g.compile(phase=PHASE_TRAIN)
    solver.load(g)
    log("Starting optimization phase 1/3", LOG_LEVEL_INFO)
    solver.optimize(1000, print_freq=1, train_input=(train_x.astype(np.float32), train_y), batch_size=batch_size)
    log("Saving final model", LOG_LEVEL_INFO)
    g.save(model_file)
    """

    """
    log("Testing inference", LOG_LEVEL_INFO)
    sample = train_x[4]
    # Deactivate any dropouts
    Dropout.set_dp_off()
    print g.infer([sample.reshape((1, 3, 240, 320)).astype(np.float32)])
    """





