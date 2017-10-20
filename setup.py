#!/usr/bin/env python

from distutils.core import setup

setup(
    name = 'TNpy',
    packages = ['TNpy', 'TNpy.data'],
    package_dir = {'TNpy': 'src'},
    description = 'Tensor Network Algorithms',
    author = 'tanlin'
)
