import pulsarbat

project = "pulsarbat"
copyright = "2022, Nikhil Mahajan"
author = "Nikhil Mahajan"

# The full version, including alpha/beta/rc tags
release = pulsarbat.__version__


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
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "page_sidebar_items": ["search-field", "page-toc", "edit-this-page"],
}

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
    "index": [],
    "install": [],
    "tutorial": [],
    "contributing": [],
    "changelog": [],
}

html_context = {
    "default_mode": "light",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    'custom.css',
]
