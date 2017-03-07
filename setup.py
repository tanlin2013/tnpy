#!/usr/bin/env python

from distutils.core import setup

setup(
    name = 'TNpy',
    packages = ['TNpy', 'TNpy.data_analysis'],
    package_dir = {'': 'src'},
    description = 'Tensor Network Algorithms',
    author = 'tanlin'
)
