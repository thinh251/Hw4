import random
import os
import tensorflow as tf
import sys
import util
import numpy as np

# python conv_tes.py P model_file_name test_folder_name
# class_letter = sys.argv[1]
# model_file_name = sys.argv[2]
# test_folder_name = sys.argv[3]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#loading test f=data
# x, y = util.load_images(test_folder_name, class_letter)
# test_x = np.array[x]
# text_y = np.array[y]
weights = []
#buidling the netowrk
tf.reset_default_graph()

def create_weight(shape):
    w = tf.Variable(tf.truncated_normal(shape))
    weights.append(w)
    print w.shape
    return w

#building the weights is hardcoded based how training behave
layers_def = util.load_layers_definition("network_description")
layer_shape= layers_def[1]
create_weight([layer_shape[0],layer_shape[0],1,layer_shape[1]])
prelayerout = layer_shape[1]
for k in layers_def.keys():
    if k > 1:
        layer_shape = layers_def[k]
        print "layer_shape ",layer_shape
        if len(layer_shape) >1:
            create_weight([layer_shape[0],layer_shape[0],prelayerout,layer_shape[1]])
            prelayerout = layer_shape[1]
        else:
            create_weight([2*layer_shape[0],layer_shape[0]])
create_weight([layer_shape[0],1])



#loading the model
saver = tf.train.Saver()
sess = tf.Session()
 # Restore variables from disk.
saver.restore(sess, "model_t3")

for i in layers_def.keys():
    wi = weights[i]
    print wi.eval()


#
# predict = tf.greater(cnn, threshold)
#         correct_pred = tf.equal(predict, tf.equal(output_holder, 1.0))
#         accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#         # Reshape images data from 3d to 4d array before
#         #  feeding into tensorflow
#         xsh = test_x.shape
#         test_x = np.reshape(test_x, (xsh[0], xsh[1], xsh[2], 1))
#         test_y = np.reshape(test_y, (len(test_y), 1))
#         # test_x = np.reshape(test_x, (len(test_x[0]), len(test_x[1]), 1))
#         accuracy = session.run(accuracy,
#                                feed_dict={input_holder: test_x,
#                                           output_holder: test_y})
#         validation_accuracy.append(accuracy)
#         print 'Validation accuracy:', np.mean(validation_accuracy)
