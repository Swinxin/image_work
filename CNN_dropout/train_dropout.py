# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:20:12 2018

@author: Vasili
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.img_provider import Portrait



import tensorflow as tf
import numpy as np




def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

height = 100
width = 100

channels = 3
n_inputs = height * width * channels

conv1_fmaps = 32
conv1_ksize = 7
conv1_stride = 2
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

pool3_fmaps = 64
n_fc1 = 64
n_outputs = 2

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32,shape=[None,128,128,3],name='X')
    training = tf.placeholder_with_default(False,shape=[],name="training")
    y = tf.placeholder(tf.int32,shape=[None],name='y')

conv1 = tf.layers.conv2d(X,filters=32, kernel_size = 5, \
                         strides=1,padding="SAME",activation=tf.nn.relu,name='conv1')

with tf.name_scope("pool1"):
    #32*32*32
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#32*32*64
conv2 = tf.layers.conv2d(pool1,filters=64,kernel_size = 3,\
                         strides=2, padding=conv2_pad,activation = tf.nn.relu,name='conv2')
with tf.name_scope("pool2"):
    #16*16*64
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    pool2_flat = tf.reshape(pool2,shape=[-1,64 * 16 * 16])
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool2_flat,64,activation=tf.nn.relu,name="fc1")
with tf.name_scope("output"):
    fc1_drop = tf.layers.dropout(fc1,0.5,training=training)
    logit = tf.layers.dense(fc1_drop,n_outputs,name="y_output")
    #小trick
    b = tf.constant(value=1,dtype=tf.float32)
    logits_eval = tf.multiply(logit,b,name='y_logit')#后面get_by_tensor
    y_prob = tf.nn.softmax(logit,name='y_prob')#这个后面没有用到,预测用

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=y)
#    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logit, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


n_epochs = 300
with tf.Session() as sess:
    init.run()
    p = Portrait(ratio=0.9,path="data")
    for epoch in range(n_epochs):
        for iteration in range(p.train.num_examples//40):
            X_batch,y_batch = p.train.next_batch(40)
            _,corr=sess.run([training_op,correct],feed_dict={X:X_batch,y:y_batch,training: True})
            prob = loss.eval(feed_dict={X:X_batch,y:y_batch})
#            print(corr)
#            print(prob)
        print("--------------------------------------------------")
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        X_test_batch,y_test_batch = p.test.next_batch(40)
        acc_test = accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})

        print(epoch, "Train accuracy:", acc_train,"test accuracy:", acc_test)
    save_path = saver.save(sess, "./model/img.pk")


#=================加载模型
'''
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("model/img.pk.meta")
    saver.restore(sess,"./model/img.pk")
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("inputs/X:0")
    y_prob = graph.get_tensor_by_name("output/y_prob:0")
    result = sess.run(y_prob,feed_dict={X:X_test_batch})
    index = tf.argmax(result,1).eval()
    print(index)
'''