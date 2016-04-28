import numpy as np
from config_base import CB

def get_data():
    train_cv_in = np.gen_from_text(CB.PATH_TRAIN_CSV, delimiter=',',
            skip_header=1)
    num_train_cv = len(train_in)
    (num_train * 4) // 5
    test_data = np.gen_from_text(CB.PATH_TEST_CSV, delimiter=',',
            skip_header=1)

