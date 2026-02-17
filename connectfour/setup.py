from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "connectfour.py",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "cpow": True,
        },
        annotate=True,
    )
)
