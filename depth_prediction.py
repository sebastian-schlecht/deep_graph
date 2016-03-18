import numpy as np, h5py
from scipy.misc import imresize

from theano.tensor.nnet import relu

from deepgraph.utils import common
from deepgraph.graph import *
from deepgraph.nn.core import *
from deepgraph.nn.conv import *
from deepgraph.nn.loss import *
from deepgraph.solver import *


def load_data(db_file):
    # Read the MAT-File into memory
    print("Loading data from database ...")
    dataset = h5py.File(db_file)

    depth_field = dataset['depths']
    depths = np.array(depth_field[0:160])

    images_field = dataset['images']
    images = np.array(images_field[0:160]).astype(np.uint8)

    # Swap axes
    images = np.swapaxes(images, 2, 3)
    depths = np.swapaxes(depths, 1, 2)

    # Resizing
    print("Resizing ...")
    img_scale = 0.5
    depth_scale = 0.125

    images_sized = np.zeros((images.shape[0],images.shape[1], int(images.shape[2]*img_scale), int(images.shape[3]*img_scale)), dtype=np.uint8)
    depths_sized = np.zeros((depths.shape[0],int(depths.shape[1]*depth_scale), int(depths.shape[2]*depth_scale)), dtype=np.float32)

    for i in range(len(images)):
        ii = imresize(images[i], img_scale)
        images_sized[i] = np.swapaxes(np.swapaxes(ii,1,2),0,1)

    # For this test, we down-sample the depth images to 64x48

    for d in range(len(depths)):
        dd = imresize(depths[d], depth_scale)
        depths_sized[d] = dd

    images = images_sized
    depths = depths_sized

    images_train, images_val = common.split_array(images, 0.9)
    depths_train, depths_val = common.split_array(depths, 0.9)
    return [(images_train, images_val),(depths_train, depths_val)]


def build_graph(batch_size):
    graph = Graph("depth_predictor")

    data            = Data(graph, "data", T.ftensor4, shape=(batch_size, 3, 240, 320))
    label           = Data(graph, "label", T.ftensor3, shape=(batch_size, 1, 60, 80))

    conv_pool_0     = Conv2DPool(graph, "conv_0", n_channels=96, kernel_shape=(11, 11), pool_size=(3, 3), activation=relu)
    lrn_0           = LRN(graph, "lrn_0")
    conv_pool_1     = Conv2DPool(graph, "conv_1", n_channels=256, kernel_shape=(5, 5), pool_size=(3, 3), activation=relu)
    lrn_1           = LRN(graph, "lrn_1")
    conv_pool_2     = Conv2DPool(graph, "conv_2", n_channels=384, kernel_shape=(3, 3), pool_size=(3, 3), activation=relu)
    lrn_2           = LRN(graph, "lrn_2")
    conv_pool_3     = Conv2DPool(graph, "conv_3", n_channels=256, kernel_shape=(3, 3), pool_size=(3, 3), activation=relu)
    flatten         = Flatten(graph, "flatten", dims=2)
    hidden_0        = FC(graph, "fc_0", n_out=4096)
    hidden_1        = FC(graph, "fc_1", n_out=4800)
    rs              = Reshape(graph, "reshape_0", shape=(batch_size, 1, 60, 80), is_output=True)

    loss            = LogarithmicScaleInvariantLoss(graph, "loss")

    # Connect
    data.connect(conv_pool_0)
    conv_pool_0.connect(lrn_0)
    lrn_0.connect(conv_pool_1)
    conv_pool_1.connect(lrn_1)
    lrn_1.connect(conv_pool_2)
    conv_pool_2.connect(lrn_2)
    lrn_2.connect(conv_pool_3)
    conv_pool_3.connect(flatten)
    flatten.connect(hidden_0)
    hidden_0.connect(hidden_1)
    hidden_1.connect(rs)
    rs.connect(loss)
    label.connect(loss)

    return graph


def print_weights(graph):
    for node in graph.nodes:
        if node.W is not None:
            print "Name: " + str(node.name)
            print node.W.get_value()


if __name__ == "__main__":
    data = load_data('./data/nyu_depth_v2_labeled.mat')
    train_x, val_x = data[0]
    train_y, val_y = data[1]

    #######################
    # Data preprocessing
    #######################
    print("Preprocessing data ...")

    # X
    # Scale into 0-1 range
    train_x = train_x.astype(np.float)
    train_x *= 0.003921
    # Subtract mean
    train_mean = np.mean(train_x, axis=0)
    idx = 0
    for element in train_x[0]:
        train_x[idx] = train_x[idx] - train_mean
        idx += 1
    # Y
    # Scale down by 100
    train_y *= (0.01)

    # Wrap data into theano shared variables
    var_train_x = common.wrap_shared(train_x)
    var_train_y = common.wrap_shared(train_y)

    var_val_x = common.wrap_shared(val_x)
    var_val_y = common.wrap_shared(val_y)

    batch_size = 128
    print("Building graph ...")
    g = build_graph(batch_size=batch_size)
    g.load_weights("data/model.zip")
    print("Compiling graph ...")
    g.compile(train_inputs=[var_train_x, var_train_y], batch_size=batch_size)
    solver = Solver(lr=0.01)
    solver.load(g)
    print("Optimizing 1/3...")
    solver.optimize(60, print_freq=100)
    print("Optimizing 2/3...")
    solver.learning_rate = 0.001
    solver.optimize(60, print_freq=100)
    print("Optimizing 3/3...")
    solver.learning_rate = 0.0001
    solver.optimize(60, print_freq=100)
    print("Saving model ...")
    g.save("data/model.zip")





