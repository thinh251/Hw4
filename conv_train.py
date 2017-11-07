import tensorflow as tf
import sys
import util


cost_mode = ['cross', 'cross-l1', 'cross-l2']

num_classes = 5
num_inputs = 25 * 25


# tf Graph input
x = tf.placeholder(tf.float32, [None, num_inputs])
y = tf.placeholder(tf.float32, [None, num_classes])

def build_conv_layer(feature_size, num_features):
    # x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    # x = tf.nn.bias_add(x, b)
    # return tf.nn.relu(x)


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

    layers = util.load_layers_definition(network_description)
    layer1 = layers[1]
