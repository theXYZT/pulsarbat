import os
import pulsarbat

project = "pulsarbat"
copyright = "2022, Nikhil Mahajan"
author = "Nikhil Mahajan"
release = pulsarbat.__version__

json_url = "https://pulsarbat.readthedocs.io/en/dev/_static/switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
if not version_match or version_match.isdigit():
    json_url = "/_static/switcher.json"
    version_match = "dev"


# -- Extensions --------------------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "numpydoc",
]

# Autosummary
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "baseband": ("https://baseband.readthedocs.io/en/latest/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# numpydoc
numpydoc_show_class_members = False


# -- General configuration ---------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_sourcelink_suffix = ""
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 2,
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "Home Page",
            "url": "https://pulsarbat.readthedocs.io/",
            "icon": "fas fa-home",
            "type": "fontawesome",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/theXYZT/pulsarbat",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "image_light": "pulsarbat_light.svg",
        "image_dark": "pulsarbat_dark.svg",
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "page_sidebar_items": ["page-toc"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}


html_context = {
    "default_mode": "light",
}

html_static_path = ["_static", ]
html_css_files = ["custom.css", ]
