import random
import os
import tensorflow as tf
import sys
import util
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cost_mode = ['cross', 'cross-l1', 'cross-l2']

batch_size = 256
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
                           strides=[1, 2, 2, 1], padding="SAME")
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


def create_cnn(layers_def):
    # Convolution Layer
    layer = None
    filter_num = 0
    channel_num = 1
    for k in layers_def.keys():
        fn = layers_def[k]
        filter_size = fn[0]
        if len(fn) > 1:  # This is not the last layer yet
            filter_num = fn[1]
            if k == 1:  # 1st layer receive input holder as input
                layer = create_conv_layer(
                    input_holder, filter_size, filter_num, channel_num)
            else:  # 2nd and above layers receive previous layer as input
                layer = create_conv_layer(layer, filter_size, filter_num,
                                          channel_num)
        else:  # Last layer in network description file
            layer = create_flatten_layer(layer)
            layer = create_fully_connected_layer(
                layer, layer.get_shape()[1:4].num_elements(), filter_size)
            layer = create_fully_connected_layer(layer, filter_size,
                                                 num_classes)
        channel_num = filter_num
    return layer


def validate_arguments(arguments):
    if len(arguments) != 8:
        print ('7 Arguments expected : <cost> <network_description> <epsilon> '
               '<max_updates> <class_letter> <model_file_name> '
               '<train_folder_name>')
        return False
    if not (arguments[1] in cost_mode):
        print 'Invalid cost, supported modes are', cost_mode
        return False
    # TODO : add more detail for argument validation
    return True


if __name__ == "__main__":
    cost = ''
    network_description = ''
    epsilon = 0.0
    max_updates = 0
    class_letter = ''
    model_file_name = ''
    train_folder_name = ''
    if validate_arguments(sys.argv):
        cost = sys.argv[1]
        network_description = sys.argv[2]
        epsilon = float(sys.argv[3])
        max_updates = int(sys.argv[4])
        class_letter = sys.argv[5]
        model_file_name = sys.argv[6]
        train_folder_name = sys.argv[7]
    else:
        sys.exit("Invalid Arguments")

    layer_def = util.load_layers_definition(network_description)
    cnn = create_cnn(layer_def)
    y_predict = tf.nn.softmax(cnn, name='y_predict')

    cost_func = tf.nn.softmax_cross_entropy_with_logits(
        logits=cnn, labels=output_holder)

    # TODO : add L1 and L2 regularization
    # if cost == cost_mode[1]: # cross entropy
    #     cost_func =
    # elif cost == cost[1]: #cross entropy with L1 regularization
    #     cost_func = tf.nn.soft
    cost = tf.reduce_mean(cost_func)
    optimizer = tf.train.GradientDescentOptimizer(epsilon).minimize(cost)
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    x, y = util.load_images(train_folder_name)
    row = len(x)
    block = row / 5
    train_accuracy = []
    validation_accuracy = []
    cost_history = []
    for i in range(5):  # 5-fold
        # Pick the current Si as the subset for testing
        sl_i = slice(i * block, (i + 1) * block)
        test_x = np.asarray(x[sl_i])
        test_y = np.asarray(y[sl_i])
        # test_y = np.split(y, [i*k, (i + 1) * k], axis=0)
        # print 'Test Y:', i, test_y
        train_x = np.delete(x, np.s_[i * block: (i + 1) * block], axis=0)
        # print 'Train X:', i, train_x
        train_y = np.delete(y, np.s_[i * block: (i + 1) * block], axis=0)
        # print 'Train Y:', i, train_y
        print 'Training on Si except S[', i, ']'
        for e in range(max_updates):  # an update is an epoch
            # shuffle the data before training
            for r in range(0, len(train_x)):
                try:
                    j = random.randint(r + 1, len(train_x) - 1)
                    if r != j:
                        train_x[r], train_x[j] = train_x[j], train_x[r]
                        train_y[r], train_y[j] = train_y[j], train_y[r]
                except ValueError:
                    pass
                    # print 'End of the list when shuffling'

            # slice the training data into mini batches and train
            for b in range(0, len(train_x), batch_size):
                batch_x = train_x[b:b + batch_size]
                batch_y = train_y[b:b + batch_size]
                # Reshape images data from 3d to 4d array before
                #  feeding into tensorflow
                sh = batch_x.shape
                batch_x = np.reshape(batch_x, (sh[0], sh[1], sh[2], 1))

                feed_data = {input_holder: batch_x, output_holder: batch_y}
                session.run(optimizer, feed_dict=feed_data)
                correct_pred = tf.equal(tf.argmax(y_predict, 1),
                                        tf.argmax(output_holder, 1))
                accuracy = tf.reduce_mean(
                    tf.cast(correct_pred, tf.float32))
                accuracy = session.run(accuracy, feed_dict=feed_data)
                train_accuracy.append(accuracy)
                cost_value = session.run(cost, feed_dict=feed_data)
                cost_history.append(cost_value)
        print 'Training accuracy:', np.mean(train_accuracy)
        print 'Training cost:', np.mean(cost_history)
        print 'Validating on S[', i, '] data'
        correct_pred = tf.equal(tf.argmax(cnn, 1),
                                tf.argmax(output_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Reshape images data from 3d to 4d array before
        #  feeding into tensorflow
        xsh = test_x.shape
        test_x = np.reshape(test_x, (xsh[0], xsh[1], xsh[2], 1))
        # test_x = np.reshape(test_x, (len(test_x[0]), len(test_x[1]), 1))
        accuracy = session.run(accuracy,
                               feed_dict={input_holder: test_x,
                                          output_holder: test_y})
        validation_accuracy.append(accuracy)
        print 'Validation accuracy:', np.mean(validation_accuracy)
        print '-------------------------------'
    # Save the weights at the last fold
    saver = tf.train.Saver()
    saver.save(session, model_file_name)

    print 'Training and validation completed'
    print 'Avg training accuracy:', np.mean(train_accuracy)
    print 'Avg validating accuracy:', np.mean(validation_accuracy)
    print 'Avg cost:', np.mean(cost_history)
