#!/usr/bin/env python

from setuptools import setup

setup(name='DeepGraph',
      version='0.1',
      description='Light-weight deep-learning API on top of Theano',
      author='Sebastian Schlecht',
      author_email='mail@sebastian-schlecht.de',
      url='https://github.com/sebastian-schlecht/deepgraph',
      packages=['deepgraph', 'deepgraph.utils', 'deepgraph.nn'],
      install_requires=[
            'Cython >= 0.20',
            'dask == 0.8.1',
            'decorator == 4.0.9',
            'graphviz == 0.4.10',
            'h5py == 2.5.0',
            'networkx == 1.11',
            'Pillow == 3.1.1',
            'pydot == 1.0.2',
            'pyparsing == 2.1.0',
            'scikit-image == 0.12.3',
            'scipy == 0.17.0',
            'six == 1.10.0',
            'Theano == 0.8.0rc1',
            'numpy >= 1.8',
            'toolz == 0.7.4'
      ],
     )