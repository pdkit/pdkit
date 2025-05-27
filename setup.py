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
VERSION = "1.4.4"


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
        'cffi==1.15.1',
        'click==8.1.3',
        'cloudpickle==2.2.0',
        'colorclass==2.2.2',
        'commonmark==0.9.1',
        'contourpy==1.0.5',
        'cycler==0.11.0',
        'dask==2022.9.2',
        'distributed==2022.9.2',
        'docopt==0.6.2',
        'fonttools==4.37.4',
        'fsspec==2022.8.2',
        'future==0.18.3',
        'HeapDict==1.0.1',
        'Jinja2==3.1.2',
        'joblib==1.2.0',
        'keras==2.10.0',
        'kiwisolver==1.4.4',
        'llvmlite==0.39.1',
        'locket==1.0.0',
        'MarkupSafe==2.1.1',
        'matplotlib==3.6.1',
        'msgpack==1.0.4',
        'numba==0.56.2',
        'numpy==1.23.5',
        'packaging==21.3',
        'pandas==1.5.0',
        'pandas-validator==0.5.0',
        'partd==1.3.0',
        'patsy==0.5.3',
        'Pillow==9.2.0',
        'praat-parselmouth==0.4.1',
        'psutil==5.9.2',
        'pybtex==0.24.0',
        'pybtex-docutils==1.0.2',
        'pycparser==2.21',
        'pyparsing==3.0.9',
        'PySoundFile==0.9.0.post1',
        'python-dateutil==2.8.2',
        'pytz==2022.4',
        'PyWavelets==1.4.1',
        'PyYAML==6.0',
        'requests==2.31.0',
        'scikit-learn==1.1.2',
        'scipy==1.15.3',
        'six==1.16.0',
        'sortedcontainers==2.4.0',
        'statsmodels==0.13.2',
        'stumpy==1.11.1',
        'threadpoolctl==3.1.0',
        'toolz==0.12.0',
        'tornado==6.1',
        'tqdm==4.64.1',
        'tsfresh==0.21.0',
        'urllib3==1.26.12',
        'zict==2.2.0',
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
