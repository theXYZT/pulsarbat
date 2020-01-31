#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['astropy>=4.0', 'numpy>=1.17']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Nikhil Mahajan",
    author_email='mahajan@astro.utoronto.ca',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description="pulsarbat (PULSAR Baseband Analysis Tools) is a Python package for analysis of radio baseband observations of pulsars.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pulsarbat',
    name='pulsarbat',
    packages=find_packages(include=['pulsarbat']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/theXYZT/pulsarbat',
    version='0.0.3',
    zip_safe=False,
)
