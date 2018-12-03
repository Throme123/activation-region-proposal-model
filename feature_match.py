#!/usr/bin/env python

# encoding: utf-8

'''
@author: zoutengtao

@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.

@file: test.py

@time: 2018/4/2 21:12

'''

import tensorflow as tf
import numpy
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import math
import pickle
import os
from tensorflow.python import pywrap_tensorflow


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)
def im2double(im):
    im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return im

def decline(inputs):
    return math.exp( -inputs )

def py_map2jpg(imgmap):
    heatmap_x = numpy.round(imgmap*255).astype(numpy.uint8)
    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)
def test():
    VGG_MEAN = [103.939, 116.779, 123.68]
    im = Image.open('crop001023.png')
    im = im.resize((512,512))
    img_matrix = numpy.asarray(im, dtype=numpy.float32) / 255
    img_matrix = numpy.reshape(img_matrix, [1, 512, 512, 3])
    input  = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 3])
    images_scaled = input * 255.0
    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=images_scaled)
    assert red.get_shape().as_list()[1:] == [512, 512, 1]
    assert green.get_shape().as_list()[1:] == [512, 512, 1]
    assert blue.get_shape().as_list()[1:] == [512, 512, 1]
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [512, 512, 3]

    with tf.variable_scope('conv1_1'):
        conv1_1_w = tf.get_variable(name='W', shape=[3, 3, 3, 64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv1_1_b = tf.get_variable(name='b', shape=[64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(bgr, conv1_1_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv1_1_b))


    with tf.variable_scope('conv1_2'):
        conv1_2_w = tf.get_variable(name='W', shape=[3, 3, 64, 64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv1_2_b = tf.get_variable(name='b', shape=[64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv1_2_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv1_2_b))
        lout = tf.nn.max_pool(lout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv2_1'):
        conv2_1_w = tf.get_variable(name='W', shape=[3, 3, 64, 128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv2_1_b = tf.get_variable(name='b', shape=[128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv2_1_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv2_1_b))

    with tf.variable_scope('conv2_2'):
        conv2_2_w = tf.get_variable(name='W', shape=[3, 3, 128, 128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv2_2_b = tf.get_variable(name='b', shape=[128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv2_2_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv2_2_b))
        lout = tf.nn.max_pool(lout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv3_1'):
        conv3_1_w = tf.get_variable(name='W', shape=[3, 3, 128, 256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv3_1_b = tf.get_variable(name='b', shape=[256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv3_1_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv3_1_b))
    with tf.variable_scope('conv3_2'):
        conv3_2_w = tf.get_variable(name='W', shape=[3, 3, 256, 256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv3_2_b = tf.get_variable(name='b', shape=[256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv3_2_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv3_2_b))
    with tf.variable_scope('conv3_3'):
        conv3_3_w = tf.get_variable(name='W', shape=[3, 3, 256, 256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv3_3_b = tf.get_variable(name='b', shape=[256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv3_3_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv3_3_b))
        lout = tf.nn.max_pool(lout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv4_1'):
        conv4_1_w = tf.get_variable(name='W', shape=[3, 3, 256, 512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv4_1_b = tf.get_variable(name='b', shape=[512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv4_1_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv4_1_b))

    with tf.variable_scope('conv4_2'):
        conv4_2_w = tf.get_variable(name='W', shape=[3, 3, 512, 512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv4_2_b = tf.get_variable(name='b', shape=[512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv4_2_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv4_2_b))

    with tf.variable_scope('conv4_3'):
        conv4_3_w = tf.get_variable(name='W', shape=[3, 3, 512, 512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv4_3_b = tf.get_variable(name='b', shape=[512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv4_3_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv4_3_b))
        lout = tf.nn.max_pool(lout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv5_1'):
        conv5_1_w = tf.get_variable(name='W', shape=[3, 3, 512, 512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv5_1_b = tf.get_variable(name='b', shape=[512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv5_1_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv5_1_b))

    with tf.variable_scope('conv5_2'):
        conv5_2_w = tf.get_variable(name='W', shape=[3, 3, 512, 512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv5_2_b = tf.get_variable(name='b', shape=[512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv5_2_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv5_2_b))

    with tf.variable_scope('conv5_3'):
        conv5_3_w = tf.get_variable(name='W', shape=[3, 3, 512, 512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv5_3_b = tf.get_variable(name='b', shape=[512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv = tf.nn.conv2d(lout, conv5_3_w, [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, conv5_3_b))
        lout = tf.nn.max_pool(lout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "VGG/cnn_net.ckpt")
        print('.........')
        conv= sess.run(lout, feed_dict={input:img_matrix})
        conv = numpy.reshape(conv,[16,16,512])
        W_ = numpy.ones([512], dtype=float)/512
        heatmap = numpy.dot(conv, W_)
        heatmap = numpy.reshape(numpy.squeeze(heatmap), [16, 16])

        curHeatMap = cv2.resize(im2double(heatmap), (256, 256))  # this line is not doing much

        curHeatMap = py_map2jpg(curHeatMap)
        image = cv2.imread('crop001023.png')
        image = cv2.resize(image, (256, 256))

        curHeatMap = im2double(image) * 0 + im2double(curHeatMap) * 0.5
        cv2.imshow("image", curHeatMap)
        cv2.waitKey(0)
        # cv2.imwrite('model2.png',curHeatMap)
        # plt.imshow(curHeatMap)
        # plt.show()


test()

