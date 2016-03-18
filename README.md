# Deep Graph
Super light-weight deep learning API based on Theano (https://github.com/Theano/Theano).

## About
Deep Graph is a very lightweight API on top of Theano inspired by similar frameworks like Keras, Lasagne or Blocks.
This library is a work in progress and not tested so far.


## Installation
Clone the repository into a folder of your choice. E.g.
````
git clone https://github.com/sebastian-schlecht/deepgraph.git && cd deepgraph
````
then run
````
python setup.py install
````
In case you are not in a virtual environment this may need sudo. Since this library is highly experimental,
I recommend to use a virtuelenv to isolate it from system-python.

### Requirements
For GPU compilation, CUDA must be installed and the CUDA compiler (nvcc) has to be a part of PATH.
See http://deeplearning.net/software/theano/tutorial/using_gpu.html for more details.

## Simple example
For a simple example have a look at test.py.