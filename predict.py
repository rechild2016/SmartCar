import math
import numpy as np
import tensorflow as tf
import cv2

data_dir = "./fig/TFrecords/traindata.tfrecords-02"  # 数据目录格式

sess = tf.InteractiveSession()  # 注册为默认session

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                   })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    height = tf.cast(features['img_height'],tf.int32)
    width = tf.cast(features['img_width'],tf.int32)
    label = tf.cast(features['label'], tf.int32)
    image = tf.reshape(image, [60, 160, 1])
    return image, label

images, labels = read_and_decode(data_dir)

min_after_dequeue = 10  # 当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别.
batch_size = 1  # 批处理大小
capacity = min_after_dequeue + 3*batch_size  # 批处理容量
images_train, labels_train = tf.train.batch([images, labels], batch_size=batch_size,
                                        capacity = capacity)
# 通过随机打乱的方式创建数据批次

image_holder = tf.placeholder(tf.float32, [batch_size, 60, 160, 1])
label_holder = tf.placeholder(tf.int32, [batch_size])
# 创建输入数据的placeholder（相当于占位符）

weight1 = variable_with_weight_loss(shape=[5, 5, 1, 32], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[32]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                        padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

weight2 = variable_with_weight_loss(shape=[5, 5, 32, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                        padding='SAME')
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value  #38400

weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

weight5 = variable_with_weight_loss(shape=[384, 3], stddev=1/384.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[3]))
logits = tf.add(tf.matmul(local3, weight5), bias5)

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

tf.global_variables_initializer().run()  # 初始化全部模型参数
tf.train.start_queue_runners()  # 启动线程（QueueRunner是一个不存在于代码中的东西，而是后台运作的一个概念）
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state("./Model/")
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("load finish!",ckpt.model_checkpoint_path)
else:
    print("load failed!")
    exit(0)

print("===========before Test========")
# 在测试集上验证精度
num_examples = 300
num_iter = int(math.ceil(num_examples / batch_size))  # math.ceil 对浮点数向上取整

step = 0
classes={"wandao":0, "shizi":0, "zhidao":0}
dict = ["wandao", "shizi", "zhidao"]

image = tf.reshape(images_train, [60, 160])
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.8
fontcolor = (0, 250, 0)

while step < num_iter:


    image_batch, label_batch = sess.run([images_train, labels_train])
    logits_value = sess.run(logits, feed_dict={image_holder: image_batch,
                                                 label_holder: label_batch})
    print("logits_value: ", logits_value[0])

    image1 = sess.run(image)

    index = logits_value[0].argmax()
    ans = (dict[index])
    print(dict[index])

    _, im_gray = cv2.threshold(image1,127, 255, cv2.THRESH_BINARY_INV)  #这里进行了反转颜色

    # img2 = cv2.Canny(im_gray[10:55,5:155], 100, 200)
    img2 = cv2.Canny(im_gray, 100, 200)

    img3 = np.ones((60,160))*255
    img4 = np.ones((60, 160)) * 255

    _, contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,offset=(0,0))

    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        print(len(contours[i]),area)
        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if(len(contours[i]) > 30 or area > 8):
            c_max.append(cnt)
    print("有效轮廓: ",len(c_max))
    cv2.drawContours(img3, c_max, -1, (0, 255, 0), 2)
    cv2.drawContours(img4, contours, -1, (0, 255, 0), 1)
    pentagram = c_max[0]
    pentagram1 = c_max[1]
    leftmost = tuple(pentagram[:, 0][pentagram[:, :, 0].argmin()])
    rightmost = tuple(pentagram1[:, 0][pentagram1[:, :, 0].argmin()])
    print(leftmost,rightmost)
    cv2.circle(img4, leftmost, 2, (0, 255, 0), 3)
    cv2.circle(img4, rightmost, 2, (0, 255, 0), 3)
    im1 = cv2.resize(im_gray, (200, 200))
    im2 = cv2.resize(img3, (200, 200))
    im3 = cv2.resize(img4, (200,200))

    cv2.putText(im3, ans, (60, 60), fontface, fontscale, fontcolor)
    cv2.imshow("img", np.hstack((im1, im3, im2)))

    cv2.waitKey(0)
    step += 1
    print("step: %d / %d\n" % (step, num_iter))

print("finish!")
