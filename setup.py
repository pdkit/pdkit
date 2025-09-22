#!/usr/bin/env python3
# Copyright 2025 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons, Alex Noble and George Roussos
# -*- coding: utf-8 -*-

import os
import re
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install


def get_version():
    with open(os.path.join("pdkit", "_version.py"), encoding="utf-8") as f:
        return re.search(r'__version__ = "([^"]+)"', f.read()).group(1)

def readme():
    with open("readme-pypi.rst", encoding="utf-8") as f:
        return f.read()


# circleci.py version
VERSION = get_version()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")
        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(
    name="pdkit",
    version=VERSION,
    description="The Parkinson`s Disease Data Science Toolkit",
    url="https://github.com/pdkit/pdkit",
    project_urls={
        "Source": "https://github.com/pdkit/pdkit",
        "Tracker": "https://github.com/pdkit/pdkit/issues",
    },
    long_description=readme(),
    long_description_content_type="text/x-rst",
    keywords="parkinson`s disease",
    author="PDkit Project members",
    author_email="g.roussos@bbk.ac.uk",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    zip_safe=False,

    # ---- Python 3.11 support ----
    python_requires=">=3.11,<3.14",

    install_requires=[
        # Core scientific stack (compatible with Python 3.11)
        # NumPy 1.23.5+ has official Python 3.11 support; keep <2 to avoid
        # downstream packages that may not yet be NumPy 2–ready.
        "numpy>=1.23.5,<2.0",            # Py3.11 supported from 1.23.5
        "pandas>=2.1,<2.4",              # pandas supports Py 3.11 (2.1–2.3)
        "scipy>=1.10",                   # SciPy supports Py 3.11

        # ML/feature extraction
        "scikit-learn>=1.3,<1.8",        # modern sklearn w/ Py3.11 wheels
        "tsfresh>=0.21.0",               # fixes for SciPy>=1.15 and NumPy>=2
        "stumpy>=1.13.0",                # depends on numba; Py3.11 now supported

        # Plotting / utils
        "matplotlib>=3.7",               # Matplotlib follows NEP29; Py3.11 OK
        "tqdm>=4.65,<5",

        # Audio + speech
        "soundfile>=0.12,<0.14",         # 0.13.x tested on Py3.11
        "praat-parselmouth>=0.4.6",

        # DL front-end (leave backend choice to user; TF/JAX/PyTorch optional)
        "keras>=3,<4",

        # Legacy helper; harmless on Py3 but kept for compatibility
        "future>=0.18.3",
    ],

    # Optional replacement for the unmaintained pandas_validator
    extras_require={
        "validation": [
            "pandera[pandas]>=0.20"      # modern DataFrame validation
        ]
    },

    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    cmdclass={
        "verify": VerifyVersionCommand,
    },
)
