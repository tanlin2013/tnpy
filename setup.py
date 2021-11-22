from setuptools import setup
# from distutils.core import setup
from tnpy import __version__


with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='tnpy',
    packages=['tnpy', 'tnpy.model'],
    package_dir={'tnpy': 'tnpy', 'tnpy.model': 'tnpy/model'},
    version=__version__,
    license='Apache 2.0',
    description='Tensor Network Algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='tao-lin',
    author_email='tanlin2013@gmail.com',
    url='https://github.com/tanlin2013/tnpy',
    download_url=f'https://github.com/tanlin2013/tnpy/archive/v{__version__}.tar.gz',
    keywords=['mps', 'tensor network'],
    install_requires=required,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)
