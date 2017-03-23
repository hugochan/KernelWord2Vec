from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("poly_word2vec_inner.pyx"),
    include_dirs=[numpy.get_include()]
)
