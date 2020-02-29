from setuptools import setup
#from distutils.core import setup
from lib import __version__


with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='TNpy',
    packages=['TNpy'],
    package_dir={'TNpy': 'lib'},
    version=__version__,
    license='MIT',
    description='Tensor Network Algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='tao-lin',
    author_email='tanlin2013@gmail.com',
    url='https://github.com/tanlin2013/TNpy',
    download_url='https://github.com/tanlin2013/TNpy/archive/v%s.tar.gz' % __version__,
    keywords=['mps', 'tensornetwork'],
    install_requires=required,
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',   # Pick a license
        'Programming Language :: Python :: 3',      # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
