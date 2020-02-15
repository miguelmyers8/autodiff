import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from setuptools import find_packages

requires = [
    'six>=1.10.0'
]

setup(
    name='autodiff',
    version='0.01',
    author='Miguel Myers',
    packages=find_packages(exclude=("test")),
    install_requires=requires,
    author_email='miguelmyers8@gmail.com',
    description='Deep Learning project',
    long_description='',
    include_package_data=True,
    zip_safe=False,
)
