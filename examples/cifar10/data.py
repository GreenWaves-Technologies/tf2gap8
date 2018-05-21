from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import pickle

import tensorflow as tf
import numpy as np


from six.moves import urllib
from six.moves import xrange

######################################################################
######################################################################

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # Silent the Warning regarding running on GPU

######################################################################
######################################################################

DATA_DIR = "/tmp/image_data"
DATA_URL ='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
IMG_SIZE = 32
NUM_CHANNEL = 3 #RGB
IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE * NUM_CHANNEL #Size of the images as a 1D array
NUM_FILES_TRAIN = 5
NUM_IMAGE_PER_FILE = 10000
NUM_IMAGE_TRAIN = NUM_FILES_TRAIN * NUM_IMAGE_PER_FILE
NUM_CLASSES = 10

######################################################################
######################################################################

def download_and_extract():
    print("")
    print("")
    print('\033[1;32m'+'#########################################################################'+'\033[1;m')
    print('\033[1;32m'+'#######################DATA CONSTRUCTION PHASE###########################'+'\033[1;m')
    print('\033[1;32m'+'#########################################################################'+'\033[1;m')
    print("")
    if not os.path.exists(DATA_DIR):
        print("")
        print('Creation of the directory')
        os.makedirs(DATA_DIR)    #If the directory does not exist it'll creat it
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):  # If there is no archive, lets download it
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
              float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    extracted_dir_path = os.path.join(DATA_DIR, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir_path):
	       tarfile.open(filepath, 'r:gz').extractall(DATA_DIR)
    else:
        print("Everything is already there")
    print("")
################################################################################

def get_file_path(filename=""):
    return os.path.join(DATA_DIR, "cifar-10-batches-py/", filename)

################################################################################

def unpickle(filename):
    file_path = get_file_path(filename)
    print("Loading data: "+ file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file,encoding='bytes' )
    print("Loading Successfull")
    return data

################################################################################

def convert_data_to_image(raw):
    """ Return the images under the form of a 4D numpy array with:
        [img_number, height, width, channel]
        """
    raw_float = np.array(raw,dtype=float) /255.0
    images = raw_float.reshape([-1, NUM_CHANNEL, IMG_SIZE, IMG_SIZE])
    images = images.transpose([0,2,3,1])
    return images

################################################################################

def load_data(filename):
    data = unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = convert_data_to_image(raw_images)
    return images, cls

################################################################################

def load_class_names():
    raw = unpickle(filename="batches.meta")[b'label_names']
    names = [x.decode('utf-8') for x in raw]
    return names

################################################################################

def load_training_data():
    images = np.zeros(shape=[NUM_IMAGE_TRAIN, IMG_SIZE, IMG_SIZE, NUM_CHANNEL], dtype = float)
    cls = np.zeros(shape=[NUM_IMAGE_TRAIN], dtype = int)
    begin = 0
    for i in range(NUM_FILES_TRAIN):
        images_batch, cls_batch = load_data(filename="data_batch_"+str(i+1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)

################################################################################

def load_test_data():
    images, cls = load_data(filename="test_batch")
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)

################################################################################

def one_hot_encoded(class_numbers,num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers)
    return np.eye(num_classes, dtype=float)[class_numbers]

################################################################################

def load_training_data_luminance():
    images = np.zeros(shape=[NUM_IMAGE_TRAIN, IMG_SIZE, IMG_SIZE, NUM_CHANNEL], dtype = float)
    cls = np.zeros(shape=[NUM_IMAGE_TRAIN], dtype = int)
    begin = 0
    for i in range(NUM_FILES_TRAIN):
        images_batch, cls_batch = load_data(filename="data_batch_"+str(i+1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end
    print("")
    print("Converting the RGB Image to Luminance Level")
    print("")
    images_lum = np.zeros(shape=[NUM_IMAGE_TRAIN, IMG_SIZE, IMG_SIZE,1], dtype = float)
    for i in range(NUM_IMAGE_TRAIN):
        for k in range(IMG_SIZE):
            for l in range(IMG_SIZE):
                images_lum[i][k][l]=0.21*images[i][k][l][0] + 0.71*images[i][k][l][1] + 0.07*images[i][k][l][2]
        sys.stdout.write('\r>> Processing  %.1f%%' % (float(i) / float( NUM_IMAGE_TRAIN) * 100.0))
        sys.stdout.flush()

    print("")
    print("")
    print("Conversion is Over")
    return images_lum, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)

################################################################################

def showing_data(images_lum, i):
    return images_lum[i]

################################################################################

################################################################################
