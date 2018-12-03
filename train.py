#!/usr/bin/env python

# encoding: utf-8

'''
@author: zoutengtao

@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.

@file: vgg_train.py

@time: 2018/11/5 20:34

'''
import tensorflow as tf
from cam import load_data as load
from cam import VGG_cam
import time



def train_run(learning_rate=0.0001, n_epoches=50, batch_size=200):
    train_data_x, train_data_y= load.load_train_data()
    batch =int(train_data_x.shape[0] / batch_size)
    print('.........bulid mode............')
    x_input = tf.placeholder(dtype=tf.float32,shape=[batch_size,160,96,3])
    y_input = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    y_input = tf.cast(y_input,tf.float32)
    model= VGG_cam.VGG16('vgg16.npy')
    model.bulid(x_input,200)
    output = model.prob
    los = -y_input*tf.log(output)-(1-y_input)*tf.log(1-output)
    loss = tf.reduce_mean(los)
    loss_train =tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    print('... training')

    start_time = time.clock()

    epoch = 0
    done_looping = False
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        while (epoch < n_epoches) and (not done_looping):
            epoch = epoch + 1
            for index in range(batch):

                iter = (epoch - 1) * batch + index
                if iter % 100 == 0:
                    print('training @ iter= %d' % (iter))
                start = index*batch_size
                end = (index+1)*batch_size
                batch_images = train_data_x[start:end]
                batch_labels = train_data_y[start:end]
                sess.run(loss_train, feed_dict={x_input:batch_images,y_input:batch_labels})

                if (iter + 1) % 8 == 0:
                    loss_ = sess.run(loss, feed_dict={x_input: batch_images, y_input: batch_labels})
                    print('loss............',loss_)
                    out_,label = sess.run([output,y_input], feed_dict={x_input:batch_images,y_input:batch_labels})
                    print('the out is ..................',out_)

        saver.save(sess,'VGG/cnn_net.ckpt')
    end_time = time.clock()

    print('Optimization complete.')

    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))


train_run()