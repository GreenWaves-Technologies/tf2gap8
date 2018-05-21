import tensorflow as tf

################################################################################

def new_weigths(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

################################################################################

def new_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

################################################################################

def new_conv_layer(input, num_channel_input, filter_size, num_filter, pooling, Relu):
    shape = [filter_size, filter_size, num_channel_input, num_filter] #Num filter == Num output
    w = new_weigths(shape)
    bias = new_bias(num_filter)
    layer = tf.nn.conv2d(input=input, filter=w, strides=[1,1,1,1], padding='VALID')
    layer += bias
    if Relu:
        layer = tf.nn.relu(layer)
    if pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return layer, w, bias

################################################################################

def new_sequence_conv_layer(input, num_channel_input, filter_size_list, num_filter_list, 
    pooling_list, Relu_list, length):
    L = []
    layer_inter = new_conv_layer(input, num_channel_input, filter_size_list[0], num_filter_list[0], pooling_list[0], Relu_list[0])[0]
    L.append(layer_inter)
    j = 1
    while (j < length):
        L.append(new_conv_layer(L[j-1], num_filter_list[j-1], filter_size_list[j], num_filter_list[j], pooling_list[j], Relu_list[j])[0])
        j = j + 1
    return L[length-1]

################################################################################

def new_fork_layer(input, num_channel_input, filter_size_list_of_list, num_filter_list_of_list, length_list, num_fork ,length_fork , pooling_list_of_list, Relu_list_of_list):
    L = []
    for i in range(num_fork):
        L.append(new_sequence_conv_layer(input, num_channel_input, filter_size_list_of_list[i], num_filter_list_of_list[i], pooling_list_of_list[i], Relu_list_of_list[i], length_list[i]))
    return tf.concat(L, 3)

################################################################################

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


################################################################################

def new_connected_layer(input, num_input, num_output, ReLu):
    w = new_weigths(shape=[num_input, num_output])
    bias = new_bias(length = num_output)
    layer = tf.matmul(input, w) + bias
    if ReLu:
        layer = tf.nn.relu(layer)
    return layer, w, bias

################################################################################
