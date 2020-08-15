from setuptools import setup, find_packages


with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='luminaire',
    version='0.1.0.dev3',

    license='Apache License 2.0',

    description='Luminaire is a python package that provides ML driven solutions for monitoring time series data',
    long_description='Luminaire provides several anomaly detection and forecasting capabilities that incorporate '
                     'correlational and seasonal patterns in the data over time as well as uncontrollable variations.',

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

    # url='',  # this is where the github pages url will go.

    # project_urls={
    # }
)
