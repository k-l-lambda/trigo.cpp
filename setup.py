"""
Setup script for Trigo C++ Python bindings

Builds the trigo_engine Python module using pybind11
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

__version__ = '1.0.0'


class get_pybind_include(object):
	"""Helper class to determine the pybind11 include path"""

	def __str__(self):
		return pybind11.get_include()


ext_modules = [
	Extension(
		'trigo_engine',
		sources=[
			'src/bindings.cpp',
			'src/trigo_game.cpp',
		],
		include_dirs=[
			get_pybind_include(),
			'include',
		],
		language='c++',
		extra_compile_args=['-std=c++17'],
	),
]


setup(
	name='trigo_engine',
	version=__version__,
	author='TrigoRL Project',
	description='Trigo 3D Go game engine with Python bindings',
	long_description='',
	ext_modules=ext_modules,
	python_requires='>=3.7',
	zip_safe=False,
)
