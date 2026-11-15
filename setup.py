"""Build script: only the Cython extensions live here.

All package metadata and dependencies are declared in pyproject.toml.
"""
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

MODULES = ["constants", "fields", "heapq", "locate", "solver", "stats"]

extensions = [
    Extension(
        f"pykonal.{name}",
        [f"pykonal/{name}.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for name in MODULES
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "cdivision": True,
            "wraparound": False,
        },
    ),
)
