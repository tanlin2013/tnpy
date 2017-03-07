#!/usr/bin/env python

from distutils.core import setup

setup(
    name = 'TNpy',
    package_dir = {'': 'src'}
    packages = ['TNpy', 'data_analysis'],
    include_dirs = [np.get_include()],
    description = 'Tensor Network Algorithms',
    author = 'tanlin'
)
