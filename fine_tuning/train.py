# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:22:02 2018

@author: VasiliShi
"""

from Inception import fetch_pretrain_model

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception
INPUT_DATA = "datasets/image_processed.npy"    
INCEPTION_PATH = os.path.join("datasets", "inception")
MODEL_PATH = "model/fine_tune_model"
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")
# 不需要从训练好的模型中加载的参数。
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数明层，在fine-tuning的过程中就是最后的全联接层。
TRAINABLE_SCOPES='InceptionV3/Logits,InceptionV3/AuxLogit'
N_CLASSES= 2
BATCH = 10
N_epochs = 10
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

fetch_pretrain_model()
reset_graph()
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_ro_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exc in exclusions:
            if var.op.name.startswith(exc):
                excluded = True
                break
        if not excluded:
            variables_ro_restore.append(var)
    return variables_ro_restore

def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variable_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        variable_to_train.extend(variables)
    return variable_to_train

processed_data = np.load(INPUT_DATA)
train_x = processed_data[0]
n_train = len(train_x)
train_y = processed_data[1]

validation_x = processed_data[2]
validation_y = processed_data[3]

testing_x = processed_data[4]
testing_y = processed_data[5]
print("traing sample:%d,Validation sample %d,Test sample %d"\
      %(n_train,len(validation_x),len(testing_x)))
with tf.name_scope("input"):
    images = tf.placeholder(tf.float32,[None,299,299,3],name="image")
    
    label = tf.placeholder(tf.int64,[None],name='label')


with tf.name_scope("train"):
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits,endpoint = inception.inception_v3(images,num_classes=2,is_training=True)
    trainable_variables = get_trainable_variables()
    
    xentropy = tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=label)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)
 
with tf.name_scope("eval"):
    correct  = tf.equal(tf.argmax(logits,1),label)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
load_fn = slim.assign_from_checkpoint_fn(INCEPTION_V3_CHECKPOINT_PATH,get_tuned_variables(),\
                                         ignore_missing_vars=True)
saver = tf.train.Saver() 

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    print("load tuned var from inception_v3 model:\033[1;35m %s \033[0m!"%INCEPTION_V3_CHECKPOINT_PATH)
    load_fn(sess)
    start = 0
    end = BATCH
    for epoch in range(N_epochs):
        #type(loss)->numpy.float32,命名为loss就报错
        losses,_,log = sess.run([loss,training_op,logits],\
                 feed_dict={images:train_x[start:end],label:train_y[start:end]})
        print(log)
        if epoch % 10 == 0 or epoch + 1 == N_epochs:
            saver.save(sess,MODEL_PATH,global_step=epoch)
            valid = sess.run(accuracy,\
                             feed_dict={images:validation_x,label:validation_y})
            print('Step %d: Training loss is %.1f Validation accuracy = %.1f%%' % (
                    epoch, losses, valid * 100.0))
        start = end  #开始新的步长
        if start == n_train:
            start = 0
        
        end = start + BATCH
        if end > n_train:
            end = n_train
    test_accuracy = sess.run(accuracy,\
                             feed_dict={images:testing_x,label:testing_y})
    print("Final test accuracy =%.1f%%"%test_accuracy)




