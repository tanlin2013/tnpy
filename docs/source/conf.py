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
from importlib import metadata

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../tnpy"))

# -- Project information -----------------------------------------------------

project = "tnpy"
copyright = "2023, Tan Tao-Lin"
author = "Tan Tao-Lin"

# The short X.Y version
version = metadata.version("tnpy")
# The full version, including alpha/beta/rc tags
release = metadata.version("tnpy")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.inheritance_diagram",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Allowing docstring in both __init__ and right under class definition
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "github_url": "https://github.com/tanlin2013/tnpy",
    "repository_url": "https://github.com/tanlin2013/tnpy",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "use_fullscreen_button": False,
    "use_download_button": False,
}

# -- Options for Sphinx autosummary ------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "autosummary": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
# autosummary_generate = True
# autosummary_imported_members = True
# numpydoc_show_class_members = False

# -- Options for Mathjax -----------------------------------------------------
# mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/node-main.min.js"
# mathjax2_config = {
#     "tex": {
#         "inlineMath": [["$", "$"], ["\\(", "\\)"]],
#         "displayMath": [["$$", "$$"]],
#         "processEscapes": True,
#     },
#     "options": {"ignoreHtmlClass": "document", "processHtmlClass": "math|output_area"}
# }

# -- Link with documentation of external projects ----------------------------
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}
