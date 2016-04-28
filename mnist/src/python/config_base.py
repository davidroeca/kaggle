import os

class CB(object):
    PATH_PYTHON = os.path.dirname(__file__)
    PATH_SRC = os.path.dirname(PATH_PYTHON)
    PATH_PROJECT = os.path.dirname(PATH_SRC)
    PATH_DATA = os.path.join(PATH_PROJECT, 'data')
    PATH_TRAIN_CSV = os.path.join(PATH_DATA, 'train.csv')
    PATH_REAL_TEST_CSV = os.path.join(PATH_DATA, 'test.csv')
    PATH_TRAIN = os.path.join(PATH_DATA, 'train.pickle')
    PATH_CV = os.path.join(PATH_DATA, 'cv.pickle')
    PATH_TEST = os.path.join(PATH_DATA, 'test.pickle')
    PATH_REAL_TEST = os.path.join(PATH_DATA, 'real_test.pickle')
