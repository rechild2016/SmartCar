import math
import time

import numpy as np
import tensorflow as tf

# import cifar10
# import cifar10_input

max_steps = 21  # 训练轮数（每一轮一个batch参与训练）
# batch_size = 100  # batch 大小
data_dir = "./fig/TFrecords/traindata.tfrecords-00"  # 数据目录格式


# 权重初始化函数
# shape：卷积核参数，格式类似于[5,5,3,32]，代表卷积核尺寸（前两个数字），通道数和卷积核个数
# stddev：标准差
# wl：L2正则化的权值参数
# 返回带有L2正则的初始化的权重参数
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    # 截断产生正态分布，就是说产生正态分布的值如果与均值的差值大于两倍的标准差，
    # 那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数
    # 与均值的差距不会超过两倍的标准差
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        # 给权重W加上L2正则，并用wl参数控制L2 loss的大小
        tf.add_to_collection('losses', weight_loss)
        # 将weight loss存在一个名为‘losses’的collection里，后面会用到
    return var

# loss计算函数
# logits：未经softmax处理过的CNN的原始输出
# labels：样本标签
# 输出：总体loss值
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)  # 类型转换为tf.int64
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    # 计算结合softmax的交叉熵（即对logits进行softmax处理，由于softmax与cross_entropy经常一起用，
    # 所以TensorFlow把他们整合到一起了），算是一个标准范式
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # 计算一个batch中交叉熵的均值
    tf.add_to_collection('losses', cross_entropy_mean)
    # 将交叉熵存在名为‘losses’的collection里

    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    # 返回total loss，total loss包括交叉熵和上面提到的weight loss

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

sess = tf.InteractiveSession()
images_train, labels_train = read_and_decode(data_dir)


min_after_dequeue = 10  # 当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别.
batch_size = 3  # 批处理大小
capacity = min_after_dequeue + 3*batch_size  # 批处理容量
image_batch, label_batch = tf.train.batch([images_train, labels_train], batch_size=batch_size, capacity=capacity)
# 通过随机打乱的方式创建数据批次


# Converting the images to a float of [0,1) to match the expected input to convolution2d
# 将图像转换为灰度值位于[0, 1)的浮点类型,以与convlution2d期望的输入匹配
float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

#  第一个卷积层

conv2d_layer_one = tf.contrib.layers.conv2d(
    float_image_batch,
    num_outputs=32,  # 生成的滤波器的数量
    kernel_size=(5, 5),  # 滤波器的高度和宽度
    activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
    stride=(2, 2),  # 对image_batch和imput_channels的跨度值
    trainable=True)
# shape(3, 125, 76,32)
# 3表示批处理数据量是3,
# 125和76表示经过卷积操作后的宽和高,这和滤波器的大小还有步长有关系

#  第一个混合/池化层,输出降采样

pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
# shape(3, 63,38,32)
# 混合层ksize,1表示选取一个批处理数据,2表示在宽的维度取2个单位,2表示在高的取两个单位,1表示选取一个滤波器也就数选择一个通道进行操作.
# strides步长表示其分别在四个维度上的跨度

# Note, the first and last dimension of the convolution output hasn't changed but the
# middle two dimensions have.
# 注意卷积输出的第一个维度和最后一个维度没有发生变化,但是中间的两个维度发生了变化


# 第二个卷积层

conv2d_layer_two = tf.contrib.layers.conv2d(
    pool_layer_one,
    num_outputs=64,  # 更多输出通道意味着滤波器数量的增加
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer,
    stride=(1, 1),
    trainable=True)
# shape(3, 63,38,64)

# 第二个混合/池化层,输出降采样

pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
# shape(3, 32, 19,64)

# 光栅化层

# 由于后面要使用softmax,因此全连接层需要修改为二阶张量,张量的第1维用于区分每幅图像,第二维用于对们每个输入张量的秩1张量
flattened_layer_two = tf.reshape(
    pool_layer_two,
    [
        batch_size,  # image_batch中的每幅图像
        -1  # 输入的其他所有维度
    ])
# 例如,如果此时一批次有三个数据的时候,则每一行就是一个数据行,然后每一列就是这个图片的数据,
# 这里的-1参数将最后一个池化层调整为一个巨大的秩1张量

# 全连接层1
hidden_layer_three = tf.contrib.layers.fully_connected(
    flattened_layer_two,
    512,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    activation_fn=tf.nn.relu
)

# 对一些神经元进行dropout操作.每个神经元以0.1的概率决定是否放电
hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

# The output of this are all the connections between the previous layers and the 120 different dog breeds
# available to train on.
# 输出是前面的层与训练中可用的120个不同品种的狗的品种的全连接
# 全连接层2
final_fully_connected = tf.contrib.layers.fully_connected(
    hidden_layer_three,
    120,  # ImageNet Dogs 数据集中狗的品种数
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1)
)


# setup-only-ignore
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=final_fully_connected, labels=label_batch))

global_step = tf.Variable(0)  # 相当于global_step,是一个全局变量,在训练完一个批次后自动增加1

#  学习率使用退化学习率的方法
# 设置初始学习率为0.01,
learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=120,
                                           decay_rate=0.95, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

# 主程序
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

coord = tf.train.Coordinator()
# 线程控制管理器
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 训练
training_steps = 1000
for step in range(training_steps):
    sess.run(optimizer)

    if step % 10 == 0:
        print("loss:", sess.run(loss))

train_prediction = tf.nn.softmax(final_fully_connected)
# setup-only-ignore
# filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
sess.close()