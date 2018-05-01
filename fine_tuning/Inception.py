# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:54:03 2018

@author: VasiliShi
"""

import cv2
import sys
import tarfile
import os
import urllib
import numpy as np
import tensorflow as tf
TF_MODELS_URL = "http://download.tensorflow.org/models"
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dowload_process(count,block_size,total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading:{}%".format(percent))
    sys.stdout.flush()
   
def fetch_pretrain_model(url = INCEPTION_V3_URL,path = INCEPTION_PATH):
    if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
        return 
#    os.mkdir(path)
    os.makedirs(path,exist_ok = True) # If the target directory already exists,ignore OSError
    tgz_path = os.path.join(path,"inception_v3.tgz")
    urllib.request.urlretrieve(url,tgz_path,dowload_process) #显示下载进度
    inception_tgz = tarfile.open(tgz_path)
    inception_tgz.extractall(path = path)
    inception_tgz.close()
    os.remove(tgz_path)

import re
CLASS_NAME_REGEX = re.compile(r"^n\d+\s+(.*)\s*$", re.M | re.U)
def load_class_names():

    clazz_path = os.path.join("datasets","inception",'imagenet_class_names.txt')
    with open(clazz_path,'rb') as f:
        content = f.read().decode("utf-8")#字节串---->字符串转换,如果是`r`就不需要进行decode
        return CLASS_NAME_REGEX.findall(content)

class_names = ["background"] + load_class_names()

from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
reset_graph()

X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(
        X, num_classes=1001, is_training=False)
predictions = end_points["Predictions"]
saver = tf.train.Saver()

img = cv2.imread("5.jpg") #input your image path
img = cv2.resize(img,(299,299))
#show(img)
img = (img - img.min()) / (img.max() - img.min()) #
#img = img.round() #shouldn't round tranform
img = img * 2 - 1
img = img.reshape(-1,299,299,3) 
with tf.Session() as sess:
    saver.restore(sess,INCEPTION_V3_CHECKPOINT_PATH)
    predictions_val = predictions.eval(feed_dict={X: img})

most_likely_class_index = np.argmax(predictions_val[0])
print("class index",most_likely_class_index)
print("class name",class_names[most_likely_class_index])

top_5 = np.argpartition(predictions_val[0], -5)[-5:]
top_5 = reversed(top_5[np.argsort(predictions_val[0][top_5])])
for i in top_5:
    print("{0}: {1:.2f}%".format(class_names[i], 100 * predictions_val[0][i]))

if __name__ == "__main__":
    fetch_pretrain_model()


