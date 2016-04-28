import numpy as np
import tensorflow as tf
import import_data
from config_base import CB
from display import display_image

def main():
    train_data, cv_data, test_data = import_data.load_train()
    sample = train_data[np.random.choice(
        train_data.shape[0], 3, replace=False), :]
    for i in sample:
        display_image(i[0], np.reshape(i[1:], [28, 28]))

if __name__ == "__main__":
    main()
