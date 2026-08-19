from setuptools import setup, find_packages

setup(
    name="bkanalysis",
    version="0.3",
    description="A module to aggregate and analyse bank accounts",
    packages=find_packages(),
    url="https://github.com/DeadEntropy/AccountReporting",
    install_requires=[
        "pandas>=3.0",
        "numpy",
        "matplotlib",
        "mortgage",
        "yfinance>=1.6.0",
        "yahooquery>=2.4.1",
        "cachetools",
    ],  # external packages as dependencies
)
