import random
import numpy as np
import tensorflow as tf
import import_data
from reformat_data import extract_labels, get_label
from config_base import CB
from display import display_image

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
    if DISPLAY_SAMPLE:
        random.sample
        sample_indeces = np.random.choice(train_data.shape[0], 3, replace=False)
        for i in sample_indeces:
            display_image(get_label(train_labels[i]),
                    np.reshape(train_data[i], [image_dim, image_dim]))
    output_size = num_labels

    ############################################################
    # Hyper Parameters
    ############################################################
    alpha = 0.005
    lmbda = 1.0
    num_steps = 1001
    batch_size = 500
    graph = tf.Graph()
    layer_sizes = [image_dim * image_dim, 100, 30, output_size]
    with graph.as_default():
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
        # Neural Network Helper Functions
        ############################################################
        def get_weights(shape):
            return tf.Variable(
                    tf.truncated_normal(shape, stddev=1/np.sqrt(shape[0])))

        def get_biases(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=1.0))

        def get_all_weights(layer_sizes):
            for i in range(1, len(layer_sizes)):
                yield get_weights(layer_sizes[i-1:i+1])

        def get_all_biases(layer_sizes):
            for i in range(1, len(layer_sizes)):
                yield get_biases([layer_sizes[i]])

        def feed_forward(weights, biases, input_layer=tf_train_data,
                training=True, activation=tf.nn.relu):
            a = input_layer
            num_weights = len(weights)
            for i, (w, b) in enumerate(zip(weights, biases)):
                z = tf.matmul(a, w) + b
                if i+1 == num_weights:
                    a = z
                else:
                    a = activation(z)
            return a

        def compute_loss(logits, labels, weights, biases, lmbda):
            unreg = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
            reg = lmbda * (np.sum(tf.nn.l2_loss(w) for w in weights) +\
                    np.sum(tf.nn.l2_loss(b) for b in biases))
            return tf.reduce_mean(unreg + reg)

        ############################################################
        # Neural Network Logic
        ############################################################
        weights = [i for i in get_all_weights(layer_sizes)]
        biases = [i for i in get_all_biases(layer_sizes)]

        logits = feed_forward(weights, biases)
        loss = compute_loss(logits, tf_train_labels, weights, biases, lmbda)

        ############################################################
        # Optimizer and Predictions
        ############################################################
        optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        cv_prediction = tf.nn.softmax(feed_forward(weights, biases,
            input_layer=tf_cv_data))
        test_prediction = tf.nn.softmax(feed_forward(weights, biases,
            input_layer=tf_test_data))

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
