"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'astropy>=4.1',
    'numpy>=1.19',
    'scipy>=1.5',
    'baseband>=4.0.2'
]

setup(
    author="Nikhil Mahajan",
    author_email='mahajan@astro.utoronto.ca',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    description=(
        'pulsarbat (PULSAR Baseband Analysis Tools) is a Python package '
        'for analysis of radio baseband observations of pulsars.'),
    install_requires=requirements,
    python_requires='>=3.8',
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pulsarbat',
    name='pulsarbat',
    packages=find_packages(include=['pulsarbat', 'pulsarbat.*']),
    url='https://github.com/theXYZT/pulsarbat',
    version='0.0.4',
    zip_safe=False,
)
