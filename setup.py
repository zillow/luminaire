from setuptools import setup, find_packages
from os import path


# Reading requirements file
with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

# Reading README.md file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='luminaire',
    version='0.4.0.dev3',

    license='Apache License 2.0',

    description='Luminaire is a python package that provides ML driven solutions for monitoring time series data',
    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Zillow Group A.I. team',
    author_email='luminaire-dev-oss@zillowgroup.com',

    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=install_requires,

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],

    keywords='anomaly-detection outlier-detection forecasting time-series automl',

    url='https://zillow.github.io/luminaire',

    project_urls={
        'Source': 'https://github.com/zillow/luminaire',
        'Tracker': 'https://github.com/zillow/luminaire/issues',
    }
)
