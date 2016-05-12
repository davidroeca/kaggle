import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

class LayerBase(object, metaclass=ABCMeta):

    @abstractmethod
    def forward_op(self, a):
        '''Must be implemented in base classes; a single feedforward step'''
        pass

    @property
    def weights(self):
        '''Represents weights of layer; if not implemented (like in case of
        a softmax or maxpool layer), set to 0 for loss computation
        '''
        if not hasattr(self, '_weights'):
            self._weights = tf.Variable(tf.constant(0.0))
        return self._weights

    @weights.setter
    def weights(self, w):
        self._weights = w

    @property
    def biases(self):
        '''Represents biases of layer; if not implemented (like in case of
        a softmax or maxpool layer), set to 0 for loss computation
        '''
        if not hasattr(self, '_biases'):
            self._biases = tf.Variable(tf.constant(0.0))
        return self._biases

    @biases.setter
    def biases(self, b):
        self._biases = b

class TfFullyConnectedLayer(LayerBase):
    '''The most basic layer for a neural network'''

    def __init__(self, activation, num_in, num_out):
        '''Instantiate the layer

        :param function activation: for neuron activation
        :param int num_in: number of inputs for layer
        :param int num_out: number of outputs for layer
        '''
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
        '''Do feedforward as simple activation(matrix multiply, add)

        :param tensor a: the input to the operation
        :returns tensor: the output of the operation
        '''
        z = tf.matmul(a, self.weights) + self.biases
        return self.activation(z)

class TfConv2dLayer(LayerBase):
    '''Specifies a 2d convolutional layer'''

    def __init__(self, activation, in_width, in_height, patch_size, depth,
                 num_channels=1, stride=[1, 2, 2, 1], padding='SAME'):
        '''Initialize the convolutional layer

        :param function activation: for neuron activation
        :param int in_width: width of the image/each channel's slice
        :param int in_height: height of the image/each channel's slice
        :param int patch_size: the width (and height) of the conv patch
        :param int depth: the number of features (output channels) computed
        :param int num_channels: the number of input channels (input depth)
        :param list(int) stride: the tf-defined stride
        :param str padding: the tf-defined padding--"SAME" or "VALID"
        '''
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
        '''Reshape a to a valid tf conv2d input and create the convolution

        :param tensor a: the input to the operation
        :returns tensor: the output of the operation
        '''
        a = tf.reshape(a, [-1, self.in_height, self.in_width, self.num_channels])
        z = tf.nn.conv2d(a, self.weights, self.stride, self.padding)
        return self.activation(z)

class TfMaxPoolLayer(LayerBase):
    '''Specifies a max pooling layer'''

    def __init__(self, kernel_size=[1, 2, 2, 1], stride_length=[1, 2, 2, 1],
                 padding='SAME'):
        '''Initialize the max pooling layer

        :param list(int) kernel_size: the tf-defined kernel size
        :param list(int) stride_length: the tf-defined stride length
        :param str padding: the tf-defined padding; either "SAME" or "VALID"
        '''
        self.kernel_size = kernel_size
        self.stride_length = stride_length
        self.padding = padding

    def forward_op(self, a):
        '''Take a convolutional output and do a max pool

        :param tensor a: the input tensor to the pooling operation
        :returns tensor: the output of the operation
        '''
        return tf.nn.max_pool(a, self.kernel_size, self.stride_length,
                              self.padding)

class TfDenselyConnectedLayer(LayerBase):
    '''Represents a layer that flattens convolutions before output'''

    def __init__(self, activation, in_width, in_height, num_in, num_out):
        '''Instantiate the layer

        :param function activation: for neuron activation
        :param int in_width: the input width of the previous convolution
        :param int in_height: the input height of the previous convolution
        :param num_in: the depth of the previous convolution
        :param num_out: the flattened output of the layer
        '''
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
        '''Take a convolutional output and flatten

        :param tensor a: the input tensor to the layer's operation
        :returns tensor: the output of the operation
        '''
        a = tf.reshape(a, [-1, self.flat_size])
        z = tf.matmul(a, self.weights) + self.biases
        return self.activation(z)

class TfSoftmaxLayer(LayerBase):
    '''Requires no inputs, but computes the softmax as a separate layer'''

    def forward_op(self, a):
        '''Takes the final output and applies softmax

        :param tensor a: the input tensor to the layer's operation
        :returns tensor: the output of the operation
        '''
        return tf.nn.softmax(a)

class Network(object):
    '''Represents a full neural network composed of instantiated layers.
    Must be instantiated within the scope of a tensorflow graph.'''

    def __init__(self, layers):
        '''Instantiate the network

        :param list(BaseLayer) layers: the layer flow of the neural network
        '''
        self.layers = layers

    def feed_forward(self, input_data):
        '''Takes input and passes it through every layer of the neural network

        :param tensor input_data: the input to the first layer
        :returns: the expected probabilities for each output classification
        '''
        a = input_data
        for layer in self.layers:
            a = layer.forward_op(a)
        return a

    def x_entropy_loss(self, tf_train_prediction, tf_train_labels, lmbda):
        '''Takes feed_forward output, the labels, and regularization to
        compute the loss

        :param tensor tf_train_prediction: the feed_forward output
        :param tensor tf_train_labels: the real labels for the output
        :param float lmbda: the regularization constant
        :returns float: the cross-entropy loss of prediction
        '''
        weights = [l.weights for l in self.layers]
        biases = [l.biases for l in self.layers]
        unreg = -tf.reduce_sum(tf_train_labels * tf.log(tf_train_prediction),
                               reduction_indices=[1])
        reg = lmbda * (np.sum(tf.nn.l2_loss(w) for w in weights) +
                            np.sum(tf.nn.l2_loss(b) for b in biases))
        return tf.reduce_mean(unreg + reg)
