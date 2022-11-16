#!/usr/bin/env python3
# Copyright 2022 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons and George Roussos
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup
from setuptools.command.install import install

# circleci.py version
VERSION = "1.4.3"


def readme():
    with open('readme-pypi.rst') as f:
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
    description='The Parkinson`s Disease Data Science Toolkit',
    url='https://github.com/pdkit/pdkit',
    long_description=readme(),
    keywords='parkinson`s disease',
    author='PDkit Project members',
    author_email='g.roussos@bbk.ac.uk',
    license='MIT',
    packages=['pdkit'],
    install_requires=[
        'numpy', 'pandas', 'scipy', 'PyWavelets', 'keras'
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
