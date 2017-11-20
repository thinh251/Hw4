import random
import os
import tensorflow as tf
import sys
import util
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

supported_modes = ['cross', 'cross-l1', 'cross-l2', 'ctest']

batch_size = 256
num_classes = 5  # 5 letters at the output
image_size = 25  # image size is 25 x 25
stride = 1
threshold = 0.5

# tf Graph input
input_holder = tf.placeholder(tf.float32, [None, image_size, image_size, 1],
                              name='i')
output_holder = tf.placeholder(tf.float32, [None, 1], name='o')
# output_holder = tf.placeholder(tf.float32, [None, num_classes], name='y')
weights = []  # Store the weights to apply L1, L2 if needed


def create_weight(shape):
    w = tf.Variable(tf.truncated_normal(shape))
    weights.append(w)
    return w


def create_bias(size):
    # TODO: generate bias in the range as described on slide Nov 13
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_conv_layer(input_layer, filter_size, num_filters, num_channels):
    w = create_weight(shape=[filter_size, filter_size,
                             num_channels, num_filters])
    biases = create_bias(num_filters)
    layer = tf.nn.conv2d(input=input_layer, filter=w,
                         strides=[1, stride, stride, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, biases)
    # Need the setting for k ?, most the documents say 2 is good
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.relu(layer)


def create_flatten_layer(layer):
    shape = layer.get_shape()
    num_features = shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fully_connected_layer(input_layer, input_size, output_size):
    w = create_weight(shape=[input_size, output_size])
    biases = create_bias(output_size)
    layer = tf.matmul(input_layer, w) + biases
    # return tf.nn.relu(layer)
    return tf.nn.sigmoid(layer)


def create_cnn(layers_def):
    # Convolution Layer
    layer = None
    filter_num = 0
    channel_num = 1  # First layer receive 1 channel from black/white images
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
            # layer = create_fully_connected_layer(layer, filter_size,
            #                                      num_classes)
            layer = create_fully_connected_layer(layer, filter_size, 1)
        channel_num = filter_num
    return layer


def graph(x, y1, y2):
    # plt.title(title)
    plt.xlabel("Max Updates")
    plt.ylabel("Cost")
    # plt.plot(x, y, color='blue', label="Training Cost", linestyle='dashed')
    training_line, = plt.plot(x, y1, color='blue',
                              label="Training Cost",
                              linestyle='dashed')
    validation_line, = plt.plot(x, y2, color='green',
                                label="Validation "
                                      "Cost",
                                linestyle='dashed')
    plt.legend(handles=[training_line, validation_line], loc=2)
    plt.show()


def validate_arguments(arguments):
    if len(arguments) != 8:
        print ('7 Arguments expected : <mode> <network_description> <epsilon> '
               '<max_updates> <class_letter> <model_file_name> '
               '<data_folder>')
        return False
    if not (arguments[1] in supported_modes):
        print 'Invalid mode, supported supported_modes are', supported_modes
        return False
    # TODO : add more detail for argument validation
    return True


def train(cost, network_description, epsilon, max_updates, class_letter,
          model_file_name, train_folder_name):
    layer_def = util.load_layers_definition(network_description)
    cnn = create_cnn(layer_def)
    # y_predict = tf.nn.softmax(cnn, name='y_predict')
    # y_predict = tf.nn.sigmoid(cnn, name='y_predict')

    # cost_func = tf.nn.softmax_cross_entropy_with_logits(
    #     logits=cnn, labels=output_holder)
    cost_func = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=cnn, labels=output_holder)

    if cost == supported_modes[1]:  # cross entropy with L1 regularization
        L1 = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
        penalty = tf.contrib.layers.apply_regularization(L1,
                                                         weights_list=weights)
        cost_func = cost_func + penalty
    elif cost == supported_modes[2]:  # cross entropy with L2 regularization
        L2 = tf.contrib.layers.l2_regularizer(scale=0.005, scope=None)
        penalty = tf.contrib.layers.apply_regularization(L2,
                                                         weights_list=weights)
        cost_func = cost_func + penalty
    cost = tf.reduce_mean(cost_func)
    optimizer = tf.train.GradientDescentOptimizer(epsilon).minimize(cost)
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    x, y = util.load_images(train_folder_name, class_letter)
    row = len(x)
    # split data set to 5 partitions for cross validation
    block = row / 5
    train_accuracy = []
    validation_accuracy = []
    cost_history = []
    cost_validation_history = []
    tg = []  # to draw training graph
    s = []  # step in training graph
    vg = []   #to draw validation graph
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
        cost_per_epoch_history = []
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
                batch_y = np.reshape(batch_y, (len(batch_y), 1))
                feed_data = {input_holder: batch_x, output_holder: batch_y}
                session.run(optimizer, feed_dict=feed_data)
                # correct_pred = tf.equal(tf.argmax(cnn, 1),
                #                         tf.argmax(output_holder, 1))
                predict = tf.greater(cnn, threshold)
                correct_pred = tf.equal(predict, tf.equal(output_holder, 1.0))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                accuracy = session.run(accuracy, feed_dict=feed_data)
                # print 'Batch accuracy:', accuracy
                train_accuracy.append(accuracy)
                cost_value = session.run(cost, feed_dict=feed_data)

                cost_history.append(cost_value)
                cost_per_epoch_history.append(cost_value)
                # if e%10 ==0 :
                #     print "max updates : "+ str(e)
                #     print 'Training mode:', np.mean(train_accuracy)
        tg.append(np.mean(cost_per_epoch_history))
        s.append((i + 1) * max_updates)
        print 'Training accuracy:', np.mean(train_accuracy)
        print 'Training mode:', np.mean(cost_history)
        print 'Validating on S[', i, '] data'
        # correct_pred = tf.equal(tf.argmax(cnn, 1),
        #                         tf.argmax(output_holder, 1))
        predict = tf.greater(cnn, threshold)
        correct_pred = tf.equal(predict, tf.equal(output_holder, 1.0))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),
                                  name="accuracy-check")
        # Reshape images data from 3d to 4d array before
        #  feeding into tensorflow
        xsh = test_x.shape
        test_x = np.reshape(test_x, (xsh[0], xsh[1], xsh[2], 1))
        test_y = np.reshape(test_y, (len(test_y), 1))
        # test_x = np.reshape(test_x, (len(test_x[0]), len(test_x[1]), 1))
        accuracy = session.run(accuracy,
                               feed_dict={input_holder: test_x,
                                          output_holder: test_y})
        validation_accuracy.append(accuracy)
        cost_validation = session.run(cost, feed_dict={input_holder: test_x,
                                                       output_holder: test_y})
        cost_validation_history.append(cost_validation)
        vg.append(i)
        print "Validation Cost: ", cost_validation
        print 'Validation accuracy:', np.mean(validation_accuracy)
        print '-------------------------------'

    # Save the weights at the last fold
    saver = tf.train.Saver()
    saver.save(session, model_file_name)

    print 'Training and validation completed'
    print 'Avg training accuracy:', np.mean(train_accuracy)
    print 'Avg Training mode:', np.mean(cost_history)
    print 'Avg Validation accuracy:', np.mean(validation_accuracy)
    print 'Avg Validation mode: ',np.mean(cost_validation_history)

    session.close()
    graph(s, tg, cost_per_epoch_history)
    return(max_updates,np.mean(cost_history),np.mean(cost_validation_history))
    # graph(vg, cost_validation_history, "Validation cost graph")


def test(network_def, model_file, test_folder, letter):
    layer_def = util.load_layers_definition(network_def)
    cnn = create_cnn(layer_def)
    predict = tf.greater(cnn, threshold)
    correct_pred = tf.equal(predict, tf.equal(output_holder, 1.0))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),
                              name="accuracy-check")
    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, model_file)

    test_x, test_y = util.load_images(test_folder, letter)
    test_x = np.asarray(test_x, dtype=float)
    test_y = np.asarray(test_y)
    xsh = test_x.shape
    test_x = np.reshape(test_x, (xsh[0], xsh[1], xsh[2], 1))
    test_y = np.reshape(test_y, (len(test_y), 1), float)

    test_accuracy = session.run(accuracy, feed_dict={input_holder: test_x,
                                                     output_holder: test_y})
    print 'Test accuracy:', test_accuracy


def test_graph(network_def, model_file, test_folder):
    # saver = tf.train.Saver()
    # init = tf.global_variables_initializer()
    # test_x, test_y = util.load_images(test_folder, by_letter=False)
    test_x, test_y = util.load_images(test_folder, letter=None)
    test_x = np.asarray(test_x, dtype=float)
    test_y = np.asarray(test_y)
    xsh = test_x.shape
    test_x = np.reshape(test_x, (xsh[0], xsh[1], xsh[2], 1))
    test_y = np.reshape(test_y, (len(test_y), 1), float)
    session = tf.Session()
    # session.run(init)
    saver = tf.train.import_meta_graph(model_file + ".meta")
    saver.restore(session, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("i:0")
    y = graph.get_tensor_by_name("o:0")
    op_to_restore = graph.get_operation_by_name("accuracy-check")
    test_accuracy = session.run(op_to_restore, feed_dict={
            x: test_x, y: test_y})
    print 'Test accuracy:', test_accuracy
    session.close()


if __name__ == "__main__":
    mode = ''
    network_description = ''
    epsilon = 0.0
    max_updates = 0
    class_letter = ''
    model_file_name = ''
    data_folder = ''
    if validate_arguments(sys.argv):
        mode = sys.argv[1]
        network_description = sys.argv[2]
        epsilon = float(sys.argv[3])
        max_updates = int(sys.argv[4])
        class_letter = sys.argv[5].strip()
        model_file_name = sys.argv[6]
        data_folder = sys.argv[7]
    else:
        sys.exit("Invalid Arguments")

    if mode == supported_modes[3]:
        test(network_description, model_file_name, data_folder, class_letter)
    else:
        train(mode, network_description, epsilon, max_updates, class_letter,
              model_file_name, data_folder)


