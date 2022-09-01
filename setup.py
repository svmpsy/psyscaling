#!/usr/bin/env python
import sys
import platform
from setuptools import setup
# from Cython.Build import cythonize


def get_version():
    with open("version.py", "rt") as f:
        return f.readline().split("=")[1].strip(' "\n')


is_cpython = platform.python_implementation() == 'CPython'
py_version = sys.version_info[:2]


extras_require = {'fast': []}
if is_cpython:
    extras_require['fast'].append("DAWG >= 0.8")
    if py_version < (3, 5):
        # lru_cache is optimized in Python 3.5
        extras_require['fast'].append("fastcache >= 1.0.2")


setup(
    name='psyscaling',
    version=get_version(),
    author='Svetlana Morozova',
    author_email='svmpsy1@gmail.com',
    url='https://github.com/svmpsy/psyscaling',

    description='Psychological scaling of text and graphic information. Used to prepare experimental data for quantitative analysis. Supports only Russian language now.',
    long_description=open('README.md').read(),

    license='MIT license',
    packages=[
        'psyscaling',
    ]
)