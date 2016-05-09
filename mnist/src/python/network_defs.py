import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

class LayerBase(object, metaclass=ABCMeta):
    @abstractmethod
    def forward_op(self, a):
        pass

class TfFullyConnectedLayer(LayerBase):

    def __init__(self, activation, num_in, num_out):
        self.activation = activation
        self.num_in = num_in
        self.num_out = num_out
        self.weights = tf.Variable(
            tf.truncated_normal((num_in, num_out),
                                stddev=1/np.sqrt(num_in)))
        if self.activation == tf.nn.relu:
            self.biases = tf.Variable(tf.truncated_normal((num_out,),
                                                          stddev=1.0))
        else:
            self.biases = tf.Variable(tf.truncated_normal((num_out,),
                                                          stddev=1.0))

    def forward_op(self, a):
        z = tf.matmul(a, self.weights) + self.biases
        return self.activation(z)

class TfConv2dLayer(LayerBase):

    def __init__(self, activation, in_width, in_height, patch_size, depth,
                 num_channels=1, stride=[1, 2, 2, 1], padding='SAME'):
        num_in = in_width * in_height * num_channels
        self.in_width = in_width
        self.in_height = in_height
        self.activation = activation
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.stride = stride
        self.padding = padding
        self.weights = tf.Variable(
            tf.truncated_normal((patch_size, patch_size, num_channels, depth),
                                stddev=1/np.sqrt(num_in)))
        if self.activation == tf.nn.relu:
            self.biases = tf.Variable(tf.constant(0.1, shape=(depth,)))
        else:
            self.biases = tf.Variable(
                tf.truncated_normal((depth,), stddev=1.0))

    def forward_op(self, a):
        a = tf.reshape(a, [-1, self.in_height, self.in_width, self.num_channels])
        z = tf.nn.conv2d(a, self.weights, self.stride, self.padding)
        return self.activation(z)

class TfMaxPoolLayer(LayerBase):

    def __init__(self, kernel_size=[1, 2, 2, 1], stride_length=[1, 2, 2, 1],
                 padding='SAME'):
        self.kernel_size = kernel_size
        self.stride_length = stride_length
        self.padding = padding

    def forward_op(self, a):
        return tf.nn.max_pool(a, self.kernel_size, self.stride_length,
                              self.padding)

class TfDenselyConnectedLayer(LayerBase):

    def __init__(self, activation, in_width, in_height, num_in, num_out):
        self.activation = activation
        self.flat_size = in_width * in_width * num_in
        self.weights = tf.Variable(
            tf.truncated_normal((self.flat_size, num_out),
                                stddev=1/np.sqrt(num_in)))
        if self.activation == tf.nn.relu:
            self.biases = tf.Variable(tf.truncated_normal((num_out,),
                                                          stddev=1.0))
        else:
            self.biases = tf.Variable(tf.truncated_normal((num_out,),
                                                          stddev=1.0))

    def forward_op(self, a):
        a = tf.reshape(a, [-1, self.flat_size])
        z = tf.matmul(a, self.weights) + self.biases
        return self.activation(z)

class TfSoftmaxLayer(LayerBase):

    def forward_op(self, a):
        return tf.nn.softmax(a)

class Network(object):

    def __init__(self, layers):
        self.layers = layers

    def feed_forward(self, input_data):
        a = input_data
        for layer in self.layers:
            a = layer.forward_op(a)
        return a

    def x_entropy_loss(self, tf_train_prediction, tf_train_labels, lmbda):
        weights = [l.weights for l in self.layers if hasattr(l, 'weights')]
        biases = [l.biases for l in self.layers if hasattr(l, 'biases')]
        unreg = -tf.reduce_sum(tf_train_labels * tf.log(tf_train_prediction),
                               reduction_indices=[1])
        reg = lmbda * (np.sum(tf.nn.l2_loss(w) for w in weights) +
                            np.sum(tf.nn.l2_loss(b) for b in biases))
        return tf.reduce_mean(unreg + reg)
