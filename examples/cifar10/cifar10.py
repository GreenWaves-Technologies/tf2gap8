import tensorflow as tf
import cnn

import data
import os
import sys
import time
import math
import numpy as np
from datetime import timedelta
import subprocess
IMG_SIZE = 32
NUM_CLASSES = 10
train_batch_size = 64
total_iterations = 0
NUM_TRAIN = 1000
NUM_IMAGE = 50000


path = os.getcwd()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

def batch_train(i):

    # Number of images in the training-set.
    num_images = len(images_train)
    x_batch = images_train[i:i+1, :, :, :]
    y_batch = labels_train[i:i+1, :]
    return x_batch, y_batch

# "Optimize" processes "num_iterations" steps on the CNN for training it on the cifar10 data set
def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()
    s=0
    for j in range(num_iterations):
        print("Phase No."+str(j+1)+" over "+str(num_iterations))
        print("")
        for i in range(NUM_IMAGE):
            x_batch, y_true_batch = batch_train(i)
            feed_dict_train = {x: x_batch,  y_true: y_true_batch}
            session.run(optimizer,  feed_dict=feed_dict_train)
            p = session.run(y_pred_cls, feed_dict = feed_dict_train)
            v = session.run(y_true_cls, feed_dict = feed_dict_train)


            if p[0]==v[0]:
                s = s  + 1 #Compteur de bonne reponse
            sys.stdout.write('\r>> Learning  %.1f%%' % (float(i) / float(NUM_IMAGE) * 100.0))
            sys.stdout.flush()
        print("")
        print("")

    print("")
    print("Efficiency: "+str((float(s)/(num_iterations*NUM_IMAGE))*100.0)+"%")
    print("")

    # Ending time.

    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# The evaluation function evaluates 
def evaluation():
    # Start-time used for printing time-usage below.
    # For the evaluation phase, the number of iterations should just be 1
    num_iterations=1
    start_time = time.time()
    s=0
    for j in range(num_iterations):
        print("Phase No."+str(j+1)+" over "+str(num_iterations))
        print("")
        for i in range(NUM_IMAGE):
            x_batch, y_true_batch = batch_train(i)
            feed_dict_train = {x: x_batch,  y_true: y_true_batch}

            p = session.run(y_pred_cls, feed_dict = feed_dict_train)
            v = session.run(y_true_cls, feed_dict = feed_dict_train)

            if p[0]==v[0]:
                s = s  + 1 # Good response counter
            sys.stdout.write('\r>> Testing  %.1f%%' % (float(i) / float(NUM_IMAGE) * 100.0))
            sys.stdout.flush()

        print("")
        print("")

    print("")
    print("Efficiency: "+str((float(s)/(num_iterations*NUM_IMAGE))*100.0)+"%")
    print("")

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


data.download_and_extract()
images_train, cls_train, labels_train = data.load_training_data_luminance()

with tf.Graph().as_default():

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape= [None, IMG_SIZE, IMG_SIZE, 1], name='x_input')
        y_true = tf.placeholder(tf.float32, shape = [None, 10], name='y_true')
        y_true_cls = tf.argmax(y_true, dimension=1)

    filter_size_1 = 5
    num_filter_1 = 8
    filter_size_2 = 5
    num_filter_2 = 12

    # Neural Network Construction
    with tf.name_scope('LAYER_1'):
        layer_conv1, weights_conv_1, bias_conv_1 = cnn.new_conv_layer(input = x, num_channel_input = 1 ,
            filter_size=filter_size_1, num_filter=num_filter_1, pooling=True, Relu = False)
    with tf.name_scope('LAYER_2'):
        layer_conv2, weights_conv_2, bias_conv_2 = cnn.new_conv_layer(input = layer_conv1, 
            num_channel_input = num_filter_1  ,filter_size=filter_size_2, num_filter=num_filter_2, 
            pooling=True, Relu = False)
    with tf.name_scope('FC_LAYER'):
        layer_flat, num_features = cnn.flatten_layer(layer_conv2)
        layer_fc1, w_fcl, b_fcl = cnn.new_connected_layer(input=layer_flat,num_input=num_features, 
            num_output=10, ReLu=True)

    with tf.name_scope('prediction'):
        y_pred = tf.nn.softmax(layer_fc1)
        y_pred_cls = tf.argmax(y_pred, dimension = 1, name="y_output")
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc1,labels=y_true)
    with tf.name_scope('COST'):
        cost = tf.reduce_mean(cross_entropy)
    with tf.name_scope('Training'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost) # Gradient Descent
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)


    ####################################################################################################################################################################################################################################
    #Learning Phase

    print("")
    print("")
    print('\033[1;32m'+'#########################################################################'+'\033[1;m')
    print('\033[1;32m'+'############################LEARNING PHASE###############################'+'\033[1;m')
    print('\033[1;32m'+'#########################################################################'+'\033[1;m')
    print("")
    print("")

    result=subprocess.run(['mkdir','-p', 'data'],stdout=subprocess.PIPE)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(sharded=True)

    # Here we choose to run the training phase for 10 steps, but it should be more to obtain a good precision
    optimize(30)
    
    #############################################################################################################################################
    #Evaluation Phase

    print("")
    print("")
    print('\033[1;32m'+'#############################cifar10##############################################'+'\033[1;m')
    print('\033[1;32m'+'############################EVALUATION PHASE###############################'+'\033[1;m')
    print('\033[1;32m'+'###########################################################################'+'\033[1;m')
    print("")
    print("")

    evaluation()
    saver.save(session, './data/model.ckpt')
    #Store the graph data 
    tf.train.write_graph(session.graph.as_graph_def(add_shapes=True), './data/', 'cifar10.pbtxt', as_text=True)
    #Save Session Graph for visualization with Tensorboard
    file_writer = tf.summary.FileWriter('logs', session.graph)
    