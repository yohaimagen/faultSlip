# run with 'build_ext --inplace' in command line


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

examples_extension = Extension(
    name="point_source",
    sources=["point_source.pyx"],
    libraries=["okada_point_source"],
    library_dirs=["."],
    include_dirs=[numpy.get_include()]
)
setup(
    name="point_source",
    ext_modules=cythonize([examples_extension])
)