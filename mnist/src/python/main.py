import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from config_base import CB
import import_data


def main():
    train_data, cv_data, test_data = import_data.load_train()
    sample = train_data[np.random.choice(
        train_data.shape[0], 3, replace=False), :]
    for i in sample:
        plt.imshow(np.reshape(i[1:], [28, 28]), cmap=cm.binary)
        plt.show(block=False)
        input('Press Enter')

if __name__ == "__main__":
    main()
