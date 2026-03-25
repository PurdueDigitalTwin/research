# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "research"
copyright = "2026, Purdue Digital Twin Lab"
author = "Juanwu Lu"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_logo = "_static/ENG_PDT_Full-RGB-Light.png"
html_favicon = "_static/ENG_PDT_Full-RGB-Light.png"

html_theme_options = {
    "logo": {
        "image_light": "_static/ENG_PDT_Full-RGB-Light.png",
    },
    "repository_url": "https://github.com/PurdueDigitalTwin/research",
    "use_repository_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

html_title = "PDT Research"
