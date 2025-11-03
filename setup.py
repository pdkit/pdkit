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
import subprocess
import sys
import shutil
import platform
import glob

from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext as _build_ext

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


class BuildExtOptional(_build_ext):
    """
    Attempt to build C extensions, but continue if they fail.
    This allows libeemd to be optional.
    """
    def run(self):
        # Build libeemd
        try:
            self.build_libeemd()
            print("libeemd C extension built successfully")
        except Exception as e:
            print(f"Warning: Could not build libeemd: {e}", file=sys.stderr)
            print("Falling back to pure Python EMD (slower)", file=sys.stderr)
        
        # Build libclose_ret
        try:
            self.build_libclose_ret()
            print("libclose_ret C extension built successfully")
        except Exception as e:
            print(f"Warning: Could not build libclose_ret: {e}", file=sys.stderr)
            print("Falling back to pure Python RPDE (slower)", file=sys.stderr)
        
        # Continue with normal extension building (even if native libs failed)
        try:
            super().run()
        except:
            pass  # No extensions to build is fine
    
    def build_libeemd(self):
        """Build libeemd using CMake"""
        # Check CMake is available
        try:
            subprocess.check_call(['cmake', '--version'], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError("CMake not found (install with: brew install cmake)")
        
        # Directories
        source_dir = os.path.abspath(os.path.join('pdkit', 'voice_features', 'native', 'libeemd'))
        build_temp = os.path.abspath(os.path.join(self.build_temp, 'libeemd_build'))
        os.makedirs(build_temp, exist_ok=True)
        
        # Configure
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp}',
            '-DBUILD_SHARED_LIBS=ON',
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        print(f"Configuring libeemd with CMake...")
        subprocess.check_call(['cmake', source_dir] + cmake_args, cwd=build_temp)
        
        # Build
        print(f"Building libeemd...")
        subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'], cwd=build_temp)
        
        # Copy built library to voice_features/_bin/
        bin_dir = os.path.join('pdkit', 'voice_features', '_bin')
        os.makedirs(bin_dir, exist_ok=True)
        
        # Find built library
        if platform.system() == 'Windows':
            lib_pattern = '**/libeemd.dll'
        elif platform.system() == 'Darwin':
            lib_pattern = '**/libeemd.dylib'
        else:
            lib_pattern = '**/libeemd.so'
        
        libs = glob.glob(os.path.join(build_temp, lib_pattern), recursive=True)
        if not libs:
            # Also try without 'lib' prefix
            alt_pattern = lib_pattern.replace('libeemd', 'eemd')
            libs = glob.glob(os.path.join(build_temp, alt_pattern), recursive=True)
        
        if libs:
            lib_name = os.path.basename(libs[0])
            dest = os.path.join(bin_dir, lib_name)
            shutil.copy2(libs[0], dest)
            print(f"Copied {lib_name} to {bin_dir}")
        else:
            raise RuntimeError(f"Built library not found in {build_temp}")
    
    def build_libclose_ret(self):
        """Build libclose_ret using CMake"""
        # Check CMake is available
        try:
            subprocess.check_call(['cmake', '--version'], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError("CMake not found (install with: brew install cmake)")
        
        # Directories
        source_dir = os.path.abspath(os.path.join('pdkit', 'voice_features', 'native', 'close_ret'))
        build_temp = os.path.abspath(os.path.join(self.build_temp, 'close_ret_build'))
        os.makedirs(build_temp, exist_ok=True)
        
        # Configure
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp}',
            '-DBUILD_SHARED_LIBS=ON',
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        print(f"Configuring libclose_ret with CMake...")
        subprocess.check_call(['cmake', source_dir] + cmake_args, cwd=build_temp)
        
        # Build
        print(f"Building libclose_ret...")
        subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'], cwd=build_temp)
        
        # Copy built library to voice_features/_bin/
        bin_dir = os.path.join('pdkit', 'voice_features', '_bin')
        os.makedirs(bin_dir, exist_ok=True)
        
        # Find built library
        if platform.system() == 'Windows':
            lib_pattern = '**/libclose_ret.dll'
        elif platform.system() == 'Darwin':
            lib_pattern = '**/libclose_ret.dylib'
        else:
            lib_pattern = '**/libclose_ret.so'
        
        libs = glob.glob(os.path.join(build_temp, lib_pattern), recursive=True)
        if not libs:
            # Also try without 'lib' prefix
            alt_pattern = lib_pattern.replace('libclose_ret', 'close_ret')
            libs = glob.glob(os.path.join(build_temp, alt_pattern), recursive=True)
        
        if libs:
            lib_name = os.path.basename(libs[0])
            dest = os.path.join(bin_dir, lib_name)
            shutil.copy2(libs[0], dest)
            print(f"Copied {lib_name} to {bin_dir}")
        else:
            raise RuntimeError(f"Built library not found in {build_temp}")


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
        
        "PyWavelets>=1.3,<2.0",
        "librosa>=0.9,<1.0",
        "EMD-signal>=1.6.4,<2.0",

        "wheel>=0.38.4"

    ],

    setup_requires=[
        'cmake>=3.12'
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

    ext_modules=[
        Extension('pdkit.voice_features._libeemd_stub', sources=[])
    ],

    cmdclass={
        "build_ext": BuildExtOptional,
        "verify": VerifyVersionCommand,
    },
)
