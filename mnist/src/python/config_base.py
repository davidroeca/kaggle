import os

class CB(object):
    PATH_PYTHON = os.path.dirname(__file__)
    PATH_SRC = os.path.dirname(PATH_PYTHON)
    PATH_PROJECT = os.path.dirname(PATH_SRC)
    PATH_DATA = os.path.join(PATH_PROJECT, 'data')
    PATH_TEST_CSV = os.path.join(PATH_DATA, 'test.csv')
    PATH_TRAIN_CSV = os.path.join(PATH_DATA, 'train.csv')

