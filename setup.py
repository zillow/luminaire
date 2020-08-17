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
    version='0.1.0.dev5',

    license='Apache License 2.0',

    description='Luminaire is a python package that provides ML driven solutions for monitoring time series data',
    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Zillow Group A.I. team',
    author_email='luminaire-dev-oss@zillowgroup.com',

    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=install_requires,

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],

    keywords='AnomalyDetection AutoML Batch Streaming',

    url='https://github.com/zillow/luminaire',

    project_urls={
        'Documentation': 'https://github.com/zillow/luminaire/blob/master/README.md',
        'Source': 'https://github.com/zillow/luminaire',
        'Tracker': 'https://github.com/zillow/luminaire/issues',
    }
)
