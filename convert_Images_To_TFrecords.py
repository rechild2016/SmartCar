import os
import tensorflow as tf 
from PIL import Image  
import cv2

#图片路径
cwd = "./fig/Images/"
#文件路径
filepath = "./fig/TFrecords/"
#存放图片个数
bestnum = 1000
#第几个图片
num = 0
#第几个TFRecord文件
recordfilenum = 0
#类别
classes=["wandao", "shizi", "zhidao"]
# classes=["test"]
#tfrecords格式文件名
tfrecordfilename = ("traindata.tfrecords-%.2d" % recordfilenum)
writer= tf.python_io.TFRecordWriter(filepath+tfrecordfilename)
#类别和路径
for index,name in enumerate(classes):
    print(index)
    print(name)
    class_path = cwd + name+ '/' 
    for img_name in os.listdir(class_path): 
        num=num+1
        if num>bestnum:
            num = 1
            recordfilenum = recordfilenum + 1
            #tfrecords格式文件名
            tfrecordfilename = ("traindata.tfrecords-%.2d" % recordfilenum)
            writer= tf.python_io.TFRecordWriter(filepath+tfrecordfilename)
        img_path = class_path+img_name #每一个图片的地址
        img=Image.open(img_path,'r').convert('L')

        size = img.size
        img_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(
             features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'img_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
            'img_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
        })) 
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()
print("convet images to TFrecords finish!",num)

