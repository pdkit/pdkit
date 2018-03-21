from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='pdkit',
    version='0.1.10',
    description='Parkinson`s Disease Kit',
    url='https://github.com/pdkit/pdkit',
    long_description=readme(),
    keywords='parkinson`s disease kit',
    author='Joan S. Pons',
    author_email='joan@dcs.bbk.ac.uk',
    license='MIT',
    packages=['pdkit'],
    install_requires=[
        'numpy', 'pandas', 'scipy', 'matplotlib'
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
    ]
)
