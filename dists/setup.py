# run with 'build_ext --inplace' in command line

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("dist",
                             sources=["dist.pyx", "c_dist.c"],
                             include_dirs=[numpy.get_include()])],
)