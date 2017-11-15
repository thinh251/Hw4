import tensorflow as tf
import numpy as np
import sys
import util


cost_mode = ['cross', 'cross-l1', 'cross-l2']


num_classes = 5  # 5 letters at the output
image_size = 25  # image size is 25 x 25
stride = 1

# tf Graph input
input_holder = tf.placeholder(tf.float32, [None, image_size, image_size, 1],
                              name='x')
output_holder = tf.placeholder(tf.float32, [None, num_classes], name='y')


def create_weight(shape):
    return tf.Variable(tf.truncated_normal(shape))


def create_bias(size):
    # TODO: generate bias in the range as described on slide Nov 13
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_conv_layer(input_layer, filter_size, num_filters, num_channels):
    weights = create_weight(shape=[filter_size, filter_size,
                                   num_channels, num_filters])
    biases = create_bias(num_filters)
    layer = tf.nn.conv2d(input=input_layer, filter=weights,
                         strides=[1, stride, stride, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, biases)
    # Need the setting for k ?, most the documents say 2 is good
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                           stride=[1, stride, stride, 1], padding="SAME")
    return tf.nn.relu(layer)


def create_flatten_layer(layer):
    shape = layer.get_shape()
    # TODO: have to understand the magic here
    num_features = shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fully_connected_layer(input_layer, input_size, output_size):
    weights = create_weight(shape=[input_size, output_size])
    biases = create_bias(output_size)
    layer = tf.matmul(input_layer, weights) + biases
    return tf.nn.relu(layer)


def build_model(layers_def):
    # Convolution Layer
    layer = None
    for i in layers_def.keys():
        fn = layers_def[i]
        if len(fn) >= 1:  # This is not the last layer yet
            filter_size = fn[0]
            filter_num = fn[1]
            if i == 1:  # 1st Layer receive input holder as input
                layer = create_conv_layer(
                    input_holder, filter_size, filter_num, 1)
            else:  # 2nd and above layers receive previous layer as input
                layer = create_conv_layer(layer, filter_size, filter_num, 1)
        else:  # Last layer in network description file
            layer = create_flatten_layer(layer)
            layer = create_fully_connected_layer(layer, filter_size,
                                                 num_classes)
    return layer


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
    x = np.random.randint(3, size=(25, 25))
    model = build_model(layer_def)

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    print session.run(model)
