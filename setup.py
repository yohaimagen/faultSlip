# run with 'build_ext --inplace' in command line
import setuptools
from Cython.Distutils import build_ext
import numpy

import subprocess

def command(cmd):
    proc = subprocess.Popen(cmd, stderr=subprocess.STDOUT, shell=True)
    output, stderr = proc.communicate(input)
    status = proc.wait()
    if status:
        raise Exception("command = {0} failed with output = {1} status {2:d}\n"
                        .format(cmd, output, status))

command(['cd psokada;make'])
# command(['ls'])
# command(['cd ..'])

psokada = setuptools.Extension(
    name="psokada/okada_stress",
    sources=["psokada/okada_stress.pyx"],
    libraries=["psokada/okada_stress"],
    library_dirs=["./psokada"],
    include_dirs=[numpy.get_include()]
)

disloc = setuptools.Extension("disloc/disloc",
                             sources=["disloc/disloc.pyx", "disloc/c_disloc.c"],
                             include_dirs=[numpy.get_include()])

dists = setuptools.Extension("dists/dist",
                             sources=["dists/dist.pyx", "dists/c_dist.c"],
                             include_dirs=[numpy.get_include()])

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="faultSlip", # Replace with your own username
    version="0.0.1",
    author="Yohai Magen",
    author_email="yohaimagen@mail.tau.ac.il",
    description="this package built for geodetic fault slip inversions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yohaimagen/faultSlip",
    download_url='https://github.com/yohaimagen/faultSlip/archive/v_0.0.1.tar.gz',
install_requires=[
          'numpy', 'scipy', 'pandas', 'matplotlib', 'json'
      ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    cmdclass = {'build_ext': build_ext},
    ext_modules =[psokada, disloc, dists],
)