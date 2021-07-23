from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='c_nms',
    ext_modules=cythonize("c_nms.pyx"),
    zip_safe=False,
    include_dirs=[np.get_include()]
)