
import os
import sys
sys.path.insert(0, os.path.abspath('./src')) 

project = 'BrahmaSumm'
copyright = '2024, Balaji Viswanathan'
author = 'Balaji Viswanathan'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',  # Auto-generates documentation from docstrings
    'sphinx.ext.napoleon'  # Supports NumPy and Google style docstrings
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
