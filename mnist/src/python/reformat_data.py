import numpy as np

def extract_labels(data_set, num_labels):
    return (data_set[:,1:].astype(np.float32),
            labels_to_1_hot(data_set[:,0], num_labels))

def labels_to_1_hot(labels, num_labels):
    return (np.arange(num_labels) == labels[:,None]).astype(np.float32)

def get_label(one_hot_label):
    return np.where(one_hot_label == np.max(one_hot_label))[0]
    
def conv_reformat(data, image_height, image_width, num_channels):
    return data.reshape(-1, image_height, image_width, num_channels)

