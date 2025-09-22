#!/usr/bin/env python3
# Copyright 2022 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons and George Roussos
# -*- coding: utf-8 -*-

import os
import re
import sys

from setuptools import setup
from setuptools.command.install import install


def get_version():
    with open(os.path.join("pdkit", "_version.py")) as f:
        return re.search(r'__version__ = "([^"]+)"', f.read()).group(1)

def readme():
    with open('readme-pypi.rst') as f:
        return f.read()
    

# circleci.py version
VERSION = get_version()

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(
    name='pdkit',
    version=VERSION,
    description='The Parkinson`s Disease Data Science Toolkit',
    url='git+https://github.com/pdkit/pdkit.git#egg=proj',
    long_description=readme(),
    keywords='parkinson`s disease',
    author='PDkit Project members',
    author_email='g.roussos@bbk.ac.uk',
    license='MIT',
    packages=['pdkit'],
    install_requires=[
        "future==1.0.0",
        "keras==3.10.0",
        "matplotlib==3.10.3",
        'numpy==1.23.5',
        "pandas==2.3.0",
        "pandas_validator==0.5.0",
        "praat-parselmouth==0.4.6",
        "scikit_learn==1.7.0",
        "scipy==1.15.3",
        "soundfile==0.13.1",
        "stumpy==1.13.0",
        "tqdm==4.67.1",
        "tsfresh==0.21.0",
    ],
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
