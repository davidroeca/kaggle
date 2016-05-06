import os
import numpy as np
import pickle
from config_base import CB

def pickle_data(dataset, path):
    basepath = os.path.basename(path)
    print("Pickling {}\n...".format(basepath))
    with open(path, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    print("Pickled {}".format(basepath))

def load_pickled_data(path):
    basepath = os.path.basename(path)
    print("Loading {}\n...".format(basepath))
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    np.random.shuffle(obj)
    print("Loaded {}".format(basepath))
    return obj

def load_train(force_load=False):
    print("Importing Training Data...")
    if any([
            force_load,
            not os.path.exists(CB.PATH_TRAIN),
            not os.path.exists(CB.PATH_CV),
            not os.path.exists(CB.PATH_TEST),
    ]):
        train_in = np.genfromtxt(CB.PATH_TRAIN_CSV, delimiter=',',
                                 skip_header=1)
        num_train = len(train_in)
        train_length = (num_train * 3) // 5
        cv_length = train_length // 3
        pickle_data(train_in[:train_length], CB.PATH_TRAIN)
        pickle_data(train_in[train_length:train_length + cv_length], CB.PATH_CV)
        pickle_data(train_in[train_length + cv_length:], CB.PATH_TEST)
    else:
        print("Data already pickled; skipping")
    train_data = load_pickled_data(CB.PATH_TRAIN)
    cv_data = load_pickled_data(CB.PATH_CV)
    test_data = load_pickled_data(CB.PATH_TEST)
    print("Successfully Imported Training Data!")
    print(train_data.shape)
    print(cv_data.shape)
    print(test_data.shape)
    return (train_data, cv_data, test_data)


def load_test(force_load=False):
    print("Importing Real Test Data...")
    if any([
            force_load,
            os.path.exists(CB.PATH_REAl_TEST),
    ]):

        real_test_in = np.genfromtxt(CB.PATH_REAL_TEST_CSV, delimiter=',',
                                     skip_header=1)
        pickle_data(real_test_data, CB.PATH_REAL_TEST)
    else:
        print("Data already pickled; skipping")
    real_test_in = load_pickled_data(CB.PATH_REAL_TEST)
    print("Successfully Imported Real Test Data!")
    print(real_test_data.shape)
    return real_test_data

