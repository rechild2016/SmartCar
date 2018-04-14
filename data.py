# !/usr/bin/env python
# coding:utf-8 
from PIL import Image, ImageTk
import os, sys
import tkinter
import time
import threading
import showImage

roadType = ["zhidao", "shizi", "wandao","test"]

file_dir = "./fig/binaryData/image_zhidao2.txt"
output = "./fig/Images/%s/"%(roadType[0])

def readBuffer(input, number, pos):
    imgList = []
    try:
        file = open(input, "rb")
        counter = 0
        file.seek(1204*pos,0)
        for line in file.read(): 
            counter += 1
            imgList.append(line)
            if counter >= number:
                break   
    finally:
        if file:
            file.close()
    return imgList

def readImage(index):    
    img = readBuffer(file_dir, 1204, index)
    # print(img)
    img2=[]
    if(len(img) == 1204):
        for item in img[2:1202]:
            for i in range(8):
                if 1&(item>>(7-i)) == 1:
                    img2.append(255)
                else :
                    img2.append(1)
    return img2

images=[]
size = 240,180
i = 0
while(True):
    i += 1
    new_img = Image.new("L", (160, 60), "white")
    imgbuff = readImage(i)
    if(len(imgbuff) == 0 ):
        print("<============imgbuff error!!!=========>")
        print("total:",i)
        break
    else:
        print("img get!")
    new_img.putdata(imgbuff)
    # new_img.save(output+"%.3d.jpg" %(i))
    new_img = new_img.resize(size,Image.ANTIALIAS)
    images.append(new_img)
    
print("finish!")

showImage.show(images)
