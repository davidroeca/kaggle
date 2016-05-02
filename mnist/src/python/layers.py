import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractproperty

class LayerBase(object, metaclass=ABCMeta):
    @abstractmethod
    def forward_op(self, a):
        pass


class TfFullyConnected(LayerBase):

    def __init__(self, num_in, num_out, activation):
        self.num_in = num_in
        self.num_out = num_out
        self.weights = tf.Variable(
                tf.truncated_normal((num_in, num_out),
                    stddev=1/np.sqrt(num_in))))
        self.biases = tf.Variable(tf.truncated_normal((num_out,), stddev=1.0))
        self.activation = activation

    def forward_op(self, a):
        z = tf.matmul(a, self.weights) + self.biases)
        return self.activation(z)

class TfConv2d(LayerBase):

    def __init__(self, inp, patch_size, <F12>jknum_channels=1):
        self.weights = tf.Variable(
                tf.truncated_normal((patch_size, patch_size, num_channels, depth),
                    stddev=1/np.sqrt(num_in))))
        self.biases 


def feed_forward(input_data, layers):
    a = input_data
    for layer in layers:
        a = layer.forward_op(a)
    return a

