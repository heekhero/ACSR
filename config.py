import os

PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'datasets')

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
