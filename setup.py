#!/usr/bin/env python

from distutils.core import setup

setup(
    name = 'tnsa',
    package_dir = {'': 'src'}
    packages = ['tnsa', 'data_analysis'],
    description = 'Tensor Network State Algorithms',
    author = 'tanlin'
)
