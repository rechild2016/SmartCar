import math
import numpy as np
import tensorflow as tf
import cv2
import Image
import time
data_dir = "./fig/TFrecords/traindata.tfrecords-02"  # 数据目录格式

sess = tf.InteractiveSession()  # 注册为默认session

def variable_with_weight_loss(shape, stddev, wl, nlayer):
    with tf.name_scope('weights'):
        var = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name="w")
        tf.summary.histogram('/weights', var)
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

with tf.name_scope('layer1'):
    weight1 = variable_with_weight_loss(shape=[5, 5, 1, 32], stddev=5e-2, wl=0.0, nlayer=1)
    kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
    with tf.name_scope('biases'):
        bias1 = tf.Variable(tf.constant(0.0, shape=[32]), name = 'b')
        tf.summary.histogram("layer1" + '/biases', bias1)
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                            padding='SAME')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


with tf.name_scope('layer2'):
    weight2 = variable_with_weight_loss(shape=[5, 5, 32, 64], stddev=5e-2, wl=0.0, nlayer=2)
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                        padding='SAME')
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

with tf.name_scope('layer3'):
    weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004, nlayer=3)
    with tf.name_scope('biases'):
        bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
        tf.summary.histogram('/biases', bias3)
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
    tf.summary.histogram('/outputs', local3)

with tf.name_scope('layer4'):
    weight4 = variable_with_weight_loss(shape=[384, 3], stddev=1/384.0, wl=0.0, nlayer=4)
    bias4 = tf.Variable(tf.constant(0.0, shape=[3]))
    logits = tf.add(tf.matmul(local3, weight4), bias4)

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
classes={"wandao":0, "shizi":0, "zhidao":0}
dict = ["wandao", "shizi", "zhidao"]

num_examples = 183
num_iter = int(math.ceil(num_examples / batch_size))  # math.ceil 对浮点数向上取整
step = 0

start_time = time.time()
while step < num_iter:
    image_batch = sess.run(images_train)
    logits_value = sess.run(logits, feed_dict={image_holder: image_batch})
    index = logits_value[0].argmax()
    ans = (dict[index])

    image1 = sess.run(tf.reshape(image_batch, [60, 160]))
    img = Image.ImageProcess(image1,ans)
    cv2.imshow("img",img)
    cv2.waitKey(100)
    step += 1
duration = time.time() - start_time
print("time:",duration)

print("finish!")
