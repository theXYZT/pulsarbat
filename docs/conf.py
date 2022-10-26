import os
import pulsarbat
import warnings

project = "pulsarbat"
copyright = "2022, Nikhil Mahajan"
author = "Nikhil Mahajan"
release = pulsarbat.__version__

json_url = "https://pulsarbat.readthedocs.io/en/latest/_static/switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
if not version_match or version_match.isdigit():
    if "dev" in release or "rc" in release:
        version_match = "latest"
        json_url = "_static/switcher.json"


# -- Extensions --------------------------------------------------------------

extensions = [
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "sphinx_toggleprompt",
    "myst_nb",
]


# Plot
plot_include_source = True
plot_formats = [("png", 96)]
plot_html_show_formats = False
plot_html_show_source_link = False

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
numpydoc_xref_param_type = True
numpydoc_xref_ignore = "all"
numpydoc_xref_aliases = {
    "array-like": ":term:`array-like <array_like>`",
    # "scalar": ":term:`scalar`",
    # "array": ":term:`array`",
    # "ndarray": "numpy.ndarray",
    "dtype": "numpy.dtype",
    "Quantity": "astropy.units.Quantity",
    "Time": "astropy.time.Time",
    "Unit": "astropy.units.Unit",
    "Signal": "pulsarbat.Signal",
    "RadioSignal": "pulsarbat.RadioSignal",
    "BasebandSignal": "pulsarbat.BasebandSignal",
    "IntensitySignal": "pulsarbat.IntensitySignal",
    "FullStokesSignal": "pulsarbat.FullStokesSignal",
    "DispersionMeasure": "pulsarbat.DispersionMeasure",
}


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
    "github_url": "https://github.com/theXYZT/pulsarbat",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pulsarbat/",
            "icon": "fa-solid fa-box",
        },
    ],
    "logo": {
        "image_light": "pulsarbat_light.svg",
        "image_dark": "pulsarbat_dark.svg",
        "alt_text": "pulsarbat",
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "page_sidebar_items": ["page-toc"],
    "switcher": {"json_url": json_url, "version_match": version_match},
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

html_context = {
    "default_mode": "auto",
    "github_user": "theXYZT",
    "github_repo": "pulsarbat",
    "github_version": "master",
    "doc_path": "docs",
}

html_static_path = [
    "_static",
]


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)


extlinks = {
    "issue": ("https://github.com/theXYZT/pulsarbat/issues/%s", "#%s"),
    "pr": ("https://github.com/theXYZT/pulsarbat/pull/%s", "PR #%s"),
}

toggleprompt_offset_right = 35


myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "html_image",
    "deflist",
]

nb_execution_mode = "off"


def setup(app):
    app.add_css_file("custom.css")
