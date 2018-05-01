# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:56:25 2018

@author: Vasili
"""

import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

INPUT_DATA = "datasets/data"
OUTPUT_FILE = "datasets/image_processed.npy"
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

def create_image_lists(sess,testing_percentage,validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)] #yield a 3-tuple (dirname,dirnames,filenames)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    validation_x = []
    validation_y = []
    extensions = ['jpg', 'jpeg']#['jpg', 'jpeg', 'JPG', 'JPEG'] ignore low or upper case
    for label,sub_dir in enumerate(sub_dirs[1:]):
        dir_name = os.path.basename(sub_dir)
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,"*"+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        print("processing:",dir_name)
        i = 0
        for file_name in file_list:
            image_raw_data = gfile.FastGFile(file_name,'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image,dtype=tf.float32)
            image = tf.image.resize_images(image,[299,299])
            try:
                image_value = sess.run(image)
                i += 1
            except tf.errors.InvalidArgumentError as e:
                continue
            #divide dataset into three parts
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_x.append(image_value)
                validation_y.append(label)
            elif chance < (testing_percentage + validation_percentage):
                test_x.append(image_value)
                test_y.append(label)
            else:
                train_x.append(image_value)
                train_y.append(label)
        print("total num",i)
    state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(state)
    np.random.shuffle(train_y)
    return np.asarray([train_x,train_y,validation_x,validation_y,test_x,test_y])

def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess,5,5)
#        np.save(OUTPUT_FILE,processed_data)

if __name__ == "__main__":
    main()