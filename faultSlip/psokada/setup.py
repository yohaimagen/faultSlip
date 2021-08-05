# run with 'build_ext --inplace' in command line


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

examples_extension = Extension(
    name="okada_stress",
    sources=["okada_stress.pyx"],
    libraries=["okada_stress"],
    library_dirs=["."],
    include_dirs=[numpy.get_include()]
)
setup(
    name="okada_stress",
    ext_modules=cythonize([examples_extension])
)