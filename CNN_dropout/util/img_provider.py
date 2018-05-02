# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:43:05 2018

@author: Vasili
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
sys.path.append(".")
#from skimage import io
from .processing import read_train_img #前面这个 点  很关键


import cv2
class DataSet(object):
    def __init__(self,images,labels):
        self._index_in_epoch = 0
        
        self._images = images
        self._label = labels
        self._epochs_completed = 0
        self._num_examples = len(self._images)
    @property
    def images(self):
        return self._images
    @property
    def label(self):
        return self._label
    @property
    def num_examples(self):
        return self._num_examples

#deprecated,bug can't loop the dataset
    
    def next_batch_deprecated(self,batch_size=6):
        start = self._index_in_epoch
        if start == 0:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._label = self.label[perm0]
        if start + batch_size > self._num_examples: #这里没有添加 如果循环结束的语句
            return read_train_img(self._images[start:],self._label[start:])
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return read_train_img(self._images[start:end],self._label[start:end])
        
    def next_batch(self,batch_size = 6,shuffle = True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start ==0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._label = self.label[perm0]
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_examples = self._num_examples - start #计算剩下的
            rest_images = self._images[start:]
            rest_labels = self._label[start:]
            if shuffle:
                perm1 = np.arange(self._num_examples)
                np.random.shuffle(perm1)
                self._images = self.images[perm1]
                self._label = self.label[perm1]
            start = 0
            self._index_in_epoch = batch_size - rest_examples#从头开始
            end = self._index_in_epoch
            new_images = self._images[:end]
            new_labels = self._label[:end]
            tmp_imgs = np.concatenate((rest_images,new_images),axis = 0)
            tmp_labels = np.concatenate((rest_labels,new_labels),axis = 0)
            return read_train_img(tmp_imgs,tmp_labels)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return read_train_img(self._images[start:end],self._label[start:end])
            
        
        
        
class Portrait(object):
    def __init__(self,ratio = 0.8,path="../data"):
        self.data_dir = path
        self.train_img = None
        self.train_label = None
        self.test_img = None
        self.test_label = None
        self.get(ratio)
        self.train = DataSet(images = self.train_img,labels = self.train_label)
        self.test = DataSet(images= self.test_img,labels= self.test_label)
        
    def get(self,ratio):
        cate=[self.data_dir+os.sep+x for x in os.listdir(self.data_dir) if os.path.isdir(self.data_dir+os.sep+x)]
        imgs = []
        labels = []
        for idx,folder in zip([0,1],cate):
            for img in os.listdir(folder):
                imgs.append(os.path.join(folder,img))
                labels.append(idx)
        count = len(labels)
        imgs = np.array(imgs)
        labels = np.array(labels) #Attention
        arr = np.arange(count)
        np.random.shuffle(arr)
        imgs = imgs[arr]
        labels = labels[arr]
        
        s = int(count * ratio)
        self.train_img = imgs[:s]
        self.test_img  = imgs[s:]
        self.train_label = labels[:s]
        self.test_label = labels[s:]
        
       
        

if __name__ == "__main__":
    p = Portrait(0.8)
    t = p.train
    a ,b= t.next_batch(5)


    cv2.imshow("img.jpg",a[2])
    cv2.waitKey (0)  
    cv2.destroyAllWindows()
#    cv2.imshow('image',img[2])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()