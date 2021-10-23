# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../tnpy'))
from tnpy import __version__


# -- Sphinx Patch -----------------------------------------------------
"""
This patch is for solving the issue -- 
`make html : TypeError: add_source_parser() takes 2 positional arguments but 3 were given`
See https://github.com/sphinx-doc/sphinx/issues/7795 for more details.
"""


def monkeypatch(cls):
    """ decorator to monkey-patch methods """
    def decorator(f):
        method = f.__name__
        old_method = getattr(cls, method)
        setattr(cls, method, lambda self, *args, **kwargs: f(old_method, self, *args, **kwargs))
    return decorator


# workaround until https://github.com/miyakogi/m2r/pull/55 is merged
@monkeypatch(sphinx.registry.SphinxComponentRegistry)
def add_source_parser(_old_add_source_parser, self, *args, **kwargs):
    # signature is (parser: Type[Parser], **kwargs), but m2r expects
    # the removed (str, parser: Type[Parser], **kwargs).
    if isinstance(args[0], str):
        args = args[1:]
    return _old_add_source_parser(self, *args, **kwargs)


# -- Project information -----------------------------------------------------

project = 'tnpy'
copyright = '2021, Tan Tao-Lin'
author = 'Tan Tao-Lin'

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.inheritance_diagram',
    'm2r',
]

# Mathjax
mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
