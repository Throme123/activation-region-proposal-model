#!/usr/bin/env python

# encoding: utf-8

'''
@author: zoutengtao

@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.

@file: VGG.py

@time: 2018/11/1 14:56

'''
import tensorflow as tf
import numpy as np


VGG_MEAN = [103.939, 116.779, 123.68]

class VGG16:

    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            W = tf.Variable(name='W', initial_value=self.data_dict[name][0], dtype=tf.float32)
            b = tf.Variable(name='b', initial_value=self.data_dict[name][1], dtype=tf.float32)
            conv = tf.nn.conv2d(bottom, W, [1,1,1,1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, b))
        return lout



    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

    def bulid(self, images, batch):
        images_scaled = images*255.0
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=images_scaled)
        assert red.get_shape().as_list()[1:] == [160, 96, 1]
        assert green.get_shape().as_list()[1:] == [160, 96, 1]
        assert blue.get_shape().as_list()[1:] == [160, 96, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [160, 96, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")#160*96
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")#80*48
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")#40*24
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")#20*12
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")#10*6
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        out = tf.nn.sigmoid(self.pool5)#5*3
        out = tf.reshape(out, shape=[batch, -1])
        self.prob = tf.reduce_mean(out, axis=1)
        self.data_dict = None
        print(("build model finished......"))



