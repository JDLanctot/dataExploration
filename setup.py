#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
from distutils.core import setup

pkg_name = 'dataExploration'
author = 'Jordan D. Lanctot'
author_email = 'jordan.lanctot@torontomu.ca'

install_requires = ['numpy',
                    'more-itertools',
                    'scipy',
                    'pandas',
                    'pyyaml',
                    'seaborn',
                    'tslearn',
                    'tqdm']

if __name__ == '__main__':
    setup(
        name=pkg_name.lower(),
        description="Data Exploration Library",
        author=author,
        author_email=author_email,
        packages=setuptools.find_packages(),
        python_requires='>=3.8',
        install_requires=install_requires
)
