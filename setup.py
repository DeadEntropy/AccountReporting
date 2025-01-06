from setuptools import setup, find_packages

setup(
    name="bkanalysis",
    version="0.3",
    description="A module to aggregate and analyse bank accounts",
    packages=find_packages(),
    url="https://github.com/DeadEntropy/AccountReporting",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "mortgage",
        "yfinance",
        "yahooquery",
        "cachetools",
    ],  # external packages as dependencies
)
