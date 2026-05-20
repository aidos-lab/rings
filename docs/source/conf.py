# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RINGS"
copyright = "2025, Corinna Coupette, Jeremy Wayland, Emily Simons, Bastian Rieck"
author = "Corinna Coupette, Jeremy Wayland, Emily Simons, Bastian Rieck"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
print(sys.path)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
]
# Ensure that member functions are documented. These are sane defaults.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

templates_path = ["_templates"]
exclude_patterns = []
html_theme = "furo"
modindex_common_prefix = ["rings."]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}


# Specifies how to actually find the sources of the modules. Ensures
# that people can jump to files in the repository directly.
def linkcode_resolve(domain, info):
    # Let's frown on global imports and do everything locally as much as
    # we can.
    import sys
    import rings

    if domain != "py":
        return None
    if not info["module"]:
        return None

    # Attempt to identify the source file belonging to an `info` object.
    # This code is adapted from the Sphinx configuration of `numpy`; see
    # https://github.com/numpy/numpy/blob/main/doc/source/conf.py.
    def find_source_file(module):
        obj = sys.modules[module]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(rings.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    try:
        module = info["module"]
        source = find_source_file(module)
    except Exception:
        source = None
    root = f"https://github.com/aidos-lab/rings/tree/main/{project.lower()}/"
    if source is not None:
        fn, start, end = source
        return root + f"{fn}#L{start}-L{end}"
    else:
        return None
