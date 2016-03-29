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
            'h5py',
            'Theano == 0.8.0',
            ]
     )