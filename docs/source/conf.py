# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata
import os
import sys

RADCOMP_VERSION = importlib.metadata.version("radcomp")

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../src/"))
sys.path.insert(0, os.path.abspath("../../src/radcomp/"))
sys.path.insert(0, os.path.abspath("../../src/radcomp/dcm/"))
sys.path.insert(0, os.path.abspath("../../src/radcomp/common/"))

project = "Radcomp"
copyright = "2023, Jake Forster"
author = "Jake Forster"
release = RADCOMP_VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

# autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = []

# add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = "classic"
html_static_path = ["_static"]
