import numpy as np, h5py
from scipy.misc import imresize
from scipy.ndimage import zoom

from theano.tensor.nnet import relu

from deepgraph.utils import common
from deepgraph.graph import *
from deepgraph.nn.core import *
from deepgraph.nn.conv import *
from deepgraph.nn.loss import *
from deepgraph.solver import *
from deepgraph.utils.logging import *

batch_size = 64


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
        images_sized[i] = np.swapaxes(np.swapaxes(ii,1,2),0,1)

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
    conv_1   = Conv2DPool(
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

    

    g = build_graph()
    model_file = "data/model.zip"
    g.load_weights(model_file)
    g.compile(train_inputs=[var_train_x, var_train_y], batch_size=batch_size)
    base_lr = 0.01
    solver = Solver(lr=base_lr)
    solver.load(g)
    Dropout.set_dp_on()
    log("Starting optimization phase", LOG_LEVEL_INFO)
    for i in range(4):
        solver.optimize(1000, print_freq=40)
        log("Saving intermediate model state", LOG_LEVEL_INFO)
        g.save(model_file)
    
   
    """
    log("Testing inference", LOG_LEVEL_INFO)


    sample = train_x[4]
    # Deactivate any dropouts
    Dropout.set_dp_off()
    print g.infer([sample.reshape((1, 3, 240, 320)).astype(np.float32)])
    """





