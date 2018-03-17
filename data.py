# !/usr/bin/env python
# coding:utf-8 
from PIL import Image, ImageTk
import os, sys
import tkinter
import time
import threading

file_dir = ".\img.txt"
output = ".\out.txt"

def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.

def readBuffer(input, number, pos):
    imgList = []
    try:
        file = open(input, "rb")
        counter = 0
        file.seek(1204*pos,0)
        for line in file.read(): 
            counter+=1
            imgList.append(line)
            if counter >= number:
                break   
    finally:
        if file:
            file.close()
    return imgList
def readImage(index):    
    img = readBuffer(file_dir, 1204, index)
    img2=[]
    for item in img[2:1202]:
        for i in range(8):
            if 1&(item>>(7-i)) == 1:
                img2.append(255)
            else :
                img2.append(1)
    return img2
images=[]
size = 240,180
for i in range(167):
    new_img = Image.new("L", (160, 60), "white")
    new_img.putdata(readImage(i))
    new_img = new_img.resize(size,Image.ANTIALIAS)
    images.append(new_img)

print("finish!")

root = tkinter.Tk()
root.bind("<Button>", button_click_exit_mainloop)
root.geometry('+%d+%d' % (200,300))

old_label_image = None
for image1 in images:
    root.geometry('%dx%d' % (image1.size[0]*2,image1.size[1]*2))
    tkpi = ImageTk.PhotoImage(image1)
    label_image = tkinter.Label(root, image=tkpi)
    label_image.place(x=0,y=0,width=image1.size[0],height=image1.size[1])
  
    if old_label_image is not None:
        old_label_image.destroy()
    old_label_image = label_image
    root.update()
    time.sleep(0.1)
    # root.mainloop()# wait until user clicks the window
