import tensorflow as tf
import cnn
import os
import sys
import time
import math
import numpy as np
from datetime import timedelta
import subprocess

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Training function for a particular number of iterations: num_iterations
#It trains the network on all the images of the MNIST data set
def optimize(num_iterations):

    start_time = time.time()
    s=0
    for j in range(num_iterations):
        print("Phase No."+str(j+1)+" over "+str(num_iterations))
        print("")
        for i in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(1)
            session.run(optimizer, feed_dict={x_inter: batch_xs, y_true: batch_ys})
            p = session.run(y_pred_cls, feed_dict={x_inter: batch_xs, y_true: batch_ys})
            v = session.run(y_true_cls,feed_dict={x_inter: batch_xs, y_true: batch_ys})
            if p[0]==v[0]:
                s = s  + 1 # Good response counter
            sys.stdout.write('\r>> Learning  %.1f%%' % (float(i) / float(2000) * 100.0))
            sys.stdout.flush()

        print("")
        print("")

    print("")
    print("Efficiency: "+str((float(s)/(num_iterations*2000))*100.0)+"%")
    print("")

    # Ending time.

    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

#Evaluation function

def test():
    # Evaluation needs only one iteration. 
    num_iterations = 1
    start_time = time.time()
    s=0
    for j in range(num_iterations):
        print("Phase No."+str(j+1)+" over "+str(num_iterations))
        print("")
        for i in range(2000):
            batch_xs, batch_ys = mnist.test.next_batch(1)

            p = session.run(y_pred_cls, feed_dict={x_inter: batch_xs, y_true: batch_ys})
            v = session.run(y_true_cls,feed_dict={x_inter: batch_xs, y_true: batch_ys})
            if p[0]==v[0]:
                s = s  + 1 # Good response counter

            if i % 50 == 0:
                print("classe exp: "+ str(p[0])+ " classe th: "+ str(v[0]))
                print("")
                
        print("")
        print("")

    print("")
    print("Efficiency: "+str((float(s)/(num_iterations*2000))*100.0)+"%")
    print("")

    # Ending time.

    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Construction of the CNN for the MNIST training

x_inter = tf.placeholder(tf.float32, shape=[None, 784], name='x_inter')
x = tf.reshape(x_inter, shape=[-1,28,28,1], name='x_input')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

conv1, w1, b1 = cnn.new_conv_layer(input = x, num_channel_input = 1 ,filter_size=5, num_filter=32, pooling=True, Relu = True)
conv2, w2, b2 = cnn.new_conv_layer(input = conv1, num_channel_input = 32 ,filter_size=5, num_filter=64, pooling=True, Relu = True)
layer_flat, num_features = cnn.flatten_layer(conv2)
layer_fc1, w_fcl, b_fcl = cnn.new_connected_layer(input=layer_flat,num_input=num_features, num_output=10, ReLu=False)

y_pred = tf.nn.softmax(layer_fc1)
y_pred_cls = tf.argmax(y_pred, dimension = 1, name="y_output")
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc1,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost) # Gradient Descent
correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# Create a session and execute the training

session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver(sharded=True)
optimize(30)

# Execute the evaluation 

test()

result=subprocess.run(['mkdir','-p','data'],stdout=subprocess.PIPE)

saver.save(session, './data/model.ckpt')

# Store the graph data 
tf.train.write_graph(session.graph.as_graph_def(add_shapes=True), './data/', 'mnist.pbtxt', as_text=True)

# Save Session Graph data for visualization with TensorBoard
file_writer = tf.summary.FileWriter('logs', session.graph)