# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:37:13 2018

@author: VasiliShi
"""

import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)

def resize(img,shape=(128,128)):#shape 为img.shape
    height,width ,channel= img.shape
    if height >128 and height > 128:
        return cv2.resize(img,shape)
    if height >=128:
        s = (height - 128) //2
        img = img[s:(s+128),:,:]
    else:#小于
        s1 = (128-height) // 2
        s2 = (128-height)-s1
        pad1 = 255 * np.ones((s1,width,channel),dtype=np.uint8)#3是通道
        pad2 = 255 * np.ones((s2,width,channel),dtype=np.uint8)#3是通道 这个地方是一个坑啊
        img = np.concatenate((pad1,img,pad2),0) #vertical
        
    if width >=128:
        s = (width - 128) //2
        img = img[:,s:(s+128),:]
    else:#小于
        s1 = (128-width) // 2
        s2 = (128-width) -s1 
        pad1 = 255 * np.ones((128,s1,channel),dtype=np.uint8)#这个int类型很关键
        pad2 = 255 * np.ones((128,s2,channel),dtype=np.uint8)#不然会出现花屏现象
        img = np.concatenate((pad1,img,pad2),1) #horizontal
    return img

def read_img_deprecated(images,label=None,isgray=False): #灰度不能够使用,有bug
    #这里面是否灰度处理是不一样的
    img_result = None
    for idx,path in enumerate(images):
        img = cv2.imread(path)#都RGB
        img = resize(img) #处理
        if isgray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(128,128,1)
        if idx == 0:
            img_result = img[np.newaxis,:,:,:] #这样写有些多余
        else:
            img_tmp = img[np.newaxis,:,:,:]
            img_result = np.concatenate((img_result,img_tmp),axis=0)
    return img_result,label

def read_img(images,gray=False):
    result = []
    for idx,path in enumerate(images):
        img = cv2.imread(path)
        img = resize(img)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(128,128,1)
        result.append(img)
    return np.array(result)
    

def read_train_img(images,label = None,gray=False,augment = False):
    result = []
    tmp_label = []
    for idx,path in enumerate(images):
        img = cv2.imread(path)
        img = resize(img)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(128,128,1)
        result.append(img) #原图
        tmp_label.append(label[idx])
        if augment:#是否数据增强
            if label[idx] == 0: #这个地方写的 [硬] 代码，后续可以改进为 ==label
                new_img = flip_h(img) # horizontal
                result.append(new_img)
                tmp_label.append(label[idx])#
                
                new_img = flip_v(img) # horizontal
                result.append(new_img)
                tmp_label.append(label[idx])#vertical
                for _ in range(4):
                    angle = np.int(np.random.uniform(-1,1) * 45)
                    new_img = rotate(img,angle)
                    result.append(new_img)
                    tmp_label.append(label[idx])
                    
    result = np.array(result)
    label = np.array(tmp_label)
    return result,label


#图片的变换
def translate(img,x,y): #移动图片
    M = np.float32([[1,0,x],[0,1,y]])
    shifted = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return shifted

def rotate(img,angle,center = None,scale = 0.9,bv=(255,255,255)):          #带黑边的旋转
    (h,w) = img.shape[:2]
    if center is None:
        center = (w//2,h//2)
    M = cv2.getRotationMatrix2D(center,angle,scale)
    rotated = cv2.warpAffine(img,M,(w,h),borderValue=bv) #白色填充
    return rotated


def rotate_bound(img,angle,bv=(255,255,255)):
    (h,w) = img.shape[:2]
    (cX,cY) = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),-angle,0.9)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0,2] += (nW / 2) - cX
    M[1,2] += (nH / 2) -cY
    return cv2.warpAffine(img,M,(nW,nH),borderValue=bv)

def flip_h(img):
    height,width,shape = img.shape
    if shape >1:
        a = img[:,:,0]
        b = img[:,:,1]
        c = img[:,:,2]
        ah = cv2.flip(a,1)#1 水平反转 #0 垂直反转#-1 水平垂直反转
        bh = cv2.flip(b,1)
        ch = cv2.flip(c,1)
        h = np.dstack((ah, bh, ch))
    else:
        a = img[:,:,0]
        h = cv2.flip(a,1)
        h = h.reshape(height,width,1)
    return h
def flip_v(img):
    height,width,shape = img.shape
    if shape >1:
        a = img[:,:,0]
        b = img[:,:,1]
        c = img[:,:,2]
        av = cv2.flip(a,0)#1 水平反转 #0 垂直反转#-1 水平垂直反转
        bv = cv2.flip(b,0)
        cv = cv2.flip(c,0)
        v = np.dstack((av, bv, cv))
    else:
        a = img[:,:,0]
        v = cv2.flip(a,1)
        v = v.reshape(height,width,1)
    return v

if __name__ == "__main__":
    import time
#    timestamp = lambda:int(time.time() * 1000)
#    path = "../data/neg"
#    files = os.listdir("../data/neg_raw")
#    for file in files:
#        img = cv2.imread(os.path.join("../data/neg_raw",file))
#        new_img = flip_h(img) # horizontal
#        cv2.imwrite("../data/neg/%s.jpg"%timestamp(),new_img)
#            
#        new_img = flip_v(img) # horizontal
#        cv2.imwrite("../data/neg/%s.jpg"%timestamp(),new_img)
#        
#        for _ in range(4):
#            angle = np.int(np.random.uniform(-1,1) * 45)
#            new_img = rotate(img,angle)
#            cv2.imwrite("../data/neg/%s.jpg"%timestamp(),new_img)
    img = cv2.imread("11165040.jpg")
    img = resize(img)
    show(img)