from setuptools import setup, find_packages


setup(
    name='luminaire',
    version='0.0.1.dev1',

    # license='',

    description='Luminaire is an internal data quality service within Zillow Group that provides an ML driven solution '
                'for monitoring time series data',
    long_description='Luminaire provides several anomaly detection / forecasting capabilities that incorporates any '
                     'correlational / seasonal pattern in the data over time and also incorporates the '
                     'uncontrollable variations. ',
    author='Luminaire Dev, Data Governance Team, ZillowGroup',
    author_email='luminiare-dev@zillowgrooup.com',

    # url='',

    python_requires='>=3.6',
    packages=find_packages(),
    # install_requires=[''],

    # classifiers=[
    # ],

    # keywords='',

    # project_urls={
    # }
)
