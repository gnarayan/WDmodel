from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules = cythonize("WDmodel/_interpolate3d.pyx"),
    include_dirs=[numpy.get_include()]
    )
