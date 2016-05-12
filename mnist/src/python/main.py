import random
import numpy as np
import tensorflow as tf
import import_data
from reformat_data import extract_labels, get_label, conv_reformat
from config_base import CB
from display import display_image
from tf_utils.network_defs import (
    TfFullyConnectedLayer,
    TfConv2dLayer,
    TfMaxPoolLayer,
    TfDenselyConnectedLayer,
    TfSoftmaxLayer,
    Network,
)

DISPLAY_SAMPLE = False

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def main():
    num_labels = 10
    train_data_lab, cv_data_lab, test_data_lab = import_data.load_train()
    train_data, train_labels = extract_labels(train_data_lab, num_labels)
    cv_data, cv_labels = extract_labels(cv_data_lab, num_labels)
    test_data, test_labels = extract_labels(test_data_lab, num_labels)
    image_dim = int(np.sqrt(train_data[0,:].shape[0]))
    num_channels = 1 # no color
    if DISPLAY_SAMPLE:
        sample_indeces = np.random.choice(train_data.shape[0], 3, replace=False)
        for i in sample_indeces:
            display_image(get_label(train_labels[i]),
                          np.reshape(train_data[i], [image_dim, image_dim]))
    output_size = num_labels

    ############################################################
    # Hyper Parameters
    ############################################################
    alpha0 = 5e-5
    lmbda = 5e-4
    num_steps = 5001
    batch_size = 100
    decay_rate = 0.98
    decay_steps = 700
    graph = tf.Graph()

    with graph.as_default():
        ############################################################
        # Network Definition
        ############################################################
        network = Network([
            TfConv2dLayer(tf.nn.relu, image_dim, image_dim, 3, 32,
                     num_channels=num_channels, stride=[1, 1, 1, 1]),
            TfMaxPoolLayer(),
            TfConv2dLayer(tf.nn.relu, 14, 14, 5, 64,
                     num_channels=32, stride=[1, 1, 1, 1]),
            TfMaxPoolLayer(),
            TfDenselyConnectedLayer(tf.nn.relu, 7, 7, 64, 1024),
            TfFullyConnectedLayer(lambda x: x, 1024, 10),
            TfSoftmaxLayer(),
        ])

        ############################################################
        # Inputs
        ############################################################
        tf_train_data = tf.placeholder(tf.float32,
                                       shape=(batch_size, image_dim * image_dim))
        tf_train_labels = tf.placeholder(tf.float32,
                                         shape=(batch_size, num_labels))
        tf_cv_data = tf.constant(cv_data)
        tf_test_data = tf.constant(test_data)
        ############################################################
        # Variables
        ############################################################
        global_step = tf.Variable(0, trainable=False)

        ############################################################
        # Neural Network Training
        ############################################################
        train_prediction = network.feed_forward(tf_train_data)
        loss = network.x_entropy_loss(train_prediction, tf_train_labels, lmbda)

        ############################################################
        # Optimizer and Predictions
        ############################################################
        alpha = tf.train.exponential_decay(alpha0, global_step, decay_steps,
                                           decay_rate)
        optimizer = tf.train.AdamOptimizer(alpha).minimize(loss,
                                                           global_step=global_step)

        ############################################################
        # Other Predictions
        ############################################################
        cv_prediction = network.feed_forward(tf_cv_data)
        test_prediction = network.feed_forward(tf_test_data)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_data[offset:offset + batch_size, :]
            batch_labels = train_labels[offset:offset + batch_size, :]
            feed_dict = { tf_train_data: batch_data,
                         tf_train_labels: batch_labels}
            _, loss_out, train_predict_out = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 100 == 0:
                print("Mini batch loss step {}: {}".format(step, loss_out))
                print("Mini batch accuracy: {}".format(
                    accuracy(train_predict_out, batch_labels)))
                print("Cross-Validation accuracy: {}".format(
                    accuracy(cv_prediction.eval(), cv_labels)))
        print("Test accuracy: {}".format(
            accuracy(test_prediction.eval(), test_labels)))

if __name__ == "__main__":
    main()
