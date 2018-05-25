#!/usr/bin/env python3
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup
from setuptools.command.install import install

# circleci.py version
VERSION = "0.4"

def readme():
    with open('README.rst') as f:
        return f.read()

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
    description='Parkinson`s Disease Kit',
    url='https://github.com/pdkit/pdkit',
    long_description=readme(),
    keywords='parkinson`s disease kit',
    author='J.S. Pons',
    author_email='joan@dcs.bbk.ac.uk',
    license='MIT',
    packages=['pdkit'],
    install_requires=[
        'numpy', 'pandas', 'scipy', 'PyWavelets'
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
