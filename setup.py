from setuptools import setup

setup(
   name='bkanalysis',
   version='0.1',
   description='A module to aggregate and analyse bank accounts',
   packages=['bkanalysis'],
   url='https://github.com/DeadEntropy/AccountReporting',
   install_requires=['pandas', 'numpy', 'matplotlib'], #external packages as dependencies
)