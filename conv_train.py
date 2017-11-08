import tensorflow as tf
import sys
import util


cost_mode = ['cross', 'cross-l1', 'cross-l2']

num_classes = 5
num_inputs = 25 * 25
stride = 1


# tf Graph input
input_holder = tf.placeholder(tf.float32, [None, num_inputs])
output_holder = tf.placeholder(tf.float32, [None, num_classes])


def initialize_weights_biases(layers):
    # layer_def is a dictionary store the network description. For example,
    # {1: [5, 4], 2:[6, 8]}. Mean that layer 1, feature size is 5, and 4 is
    # the number of features“
    if not layers:
        weights = dict()
        biases = dict()
        max_key = 0
        input_num = 1
        output_num = 0
        for k in layers.keys:
            # Get the feature size and number of features from layer
            # definition.
            fn = layers[k]
            filter_num = fn[0]
            output_num = (input_num - filter_num)/stride + 1
            if max_key < k:
                # Convolutional layer
                # weight key, and bias key for each layer
                weights[k] = tf.Variable(tf.random_normal(
                    [filter_num, filter_num, input_num, output_num]))
                biases[k] = tf.Variable(tf.random_normal([fn[1]]))
            else:
                # Dense layer
                weights['dense'] = tf.Variable(tf.random_normal([input_num, fn[0]]))
                biases['dense'] = tf.Variable(tf.random_normal([fn[0]])),
            max_key = k
            input_num = output_num
        weights['out'] = tf.Variable(tf.random_normal([output_num, num_classes]))
        biases['out'] = tf.Variable(tf.random_normal([num_classes]))
        return weights, biases


def build_conv_layer(x, w, b, feature_size, num_features):
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def build_model(x, cost, layers):
    x = tf.reshape(x, shape=[-1, 25, 25, 1])
    # Convolution Layer
    weights, biases = initialize_weights_biases(layers)
    convolutional_layer = None
    for w in weights.keys():
        if w != 'dense' or w != 'out':
            convolutional_layer = build_conv_layer(x, weights[w], biases[w])
            # Max Pooling (down-sampling)
            convolutional_layer = maxpool2d(convolutional_layer, k=2)

    full_connected = tf.reshape(convolutional_layer, [-1, weights['dense'].get_shape().as_list()[0]])
    full_connected = tf.add(tf.matmul(full_connected, weights['dense']), biases['dense'])
    full_connected = tf.nn.relu(full_connected)

    # Output, class prediction
    out = tf.add(tf.matmul(full_connected, weights['out']), biases['out'])
    return out


def validate_arguments(arguments):
    if len(arguments) < 8:
        print ('Missing arguments')
        return False
    if not (arguments[1] in cost_mode):
        print 'Invalid cost, supported modes are', cost_mode
        return False
    # TODO : add more detail for argument validation
    return True


if __name__ == "__main__":
    cost = ''
    network_description = ''
    if validate_arguments(sys.argv):
        cost = sys.argv[1]
        network_description = sys.argv[2]

    else:
        sys.exit("Invalid Arguments")

    layer_def = util.load_layers_definition(network_description)
    layer1 = layer_def[1]
