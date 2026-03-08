from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "connectfour",
        ["connectfour.py"],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-funroll-loops",
        ],
    )
]

setup(ext_modules=cythonize(extensions, annotate=True))
