import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractproperty

class LayerBase(object, metaclass=ABCMeta):
    @abstractmethod
    def forward_op(self, a):
        pass

class TfFullyConnected(LayerBase):

    def __init__(self, activation, num_in, num_out):
        self.activation = activation
        self.num_in = num_in
        self.num_out = num_out
        self.weights = tf.Variable(
            tf.truncated_normal((num_in, num_out),
                                stddev=1/np.sqrt(num_in))))
        if self.activation == tf.nn.relu:
            self.biases = tf.Variable(tf.truncated_normal((num_out,),
                                                          stddev=1.0))
        else:
            self.biases = tf.Variable(tf.truncated_normal((num_out,),
                                                          stddev=1.0))

    def forward_op(self, a):
        z = tf.matmul(a, self.weights) + self.biases)
        return self.activation(z)

class TfConv2d(LayerBase):

    def __init__(self, activation, in_width, in_height, patch_size,
                 num_channels=1, stride=[1, 2, 2, 1], padding='SAME'):
        self.activation = activation
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.stride = stride
        self.padding = padding
        self.weights = tf.Variable(
            tf.truncated_normal((patch_size, patch_size, num_channels, depth),
                                stddev=1/np.sqrt(num_in))))
        if self.activation == tf.nn.relu:
            self.biases = tf.Variable(tf.constant(0.1, shape=(depth,)))
        else:
            self.biases = tf.Variable(
                tf.truncated_normal((depth,), stddev=1.0))

    def forward_op(self, a):
        z = tf.nn.conv2d(a, self.weights, self.stride, self.padding)
        return self.activation(z)

class TfMaxPool(LayerBase):

    def __init__(self, kernel_size=[1, 2, 2, 1], stride_length=[1, 2, 2, 1],
                 padding='SAME'):
        self.kernel_size = kernel_size
        self.stride_length = stride_length

    def forward_op(self, a):
        return tf.nn.max_pool(a, self.kernel_size, self.stride_length,
                              self.padding)

class Network(object):

    def __init__(self, layers):
        self.layers = layers

    def feed_forward(self, input_data):
        a = input_data
        for layer in self.layers:
            a = layer.forward_op(a)
        return a
