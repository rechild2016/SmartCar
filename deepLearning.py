import math
import time

import numpy as np
import tensorflow as tf

import cifar10
import cifar10_input

max_steps = 21  # 训练轮数（每一轮一个batch参与训练）
batch_size = 128  # batch 大小
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'  # 数据目录


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


cifar10.maybe_download_and_extract()  # 下载数据集，并解压到默认位置

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                            batch_size=batch_size)
# 产生训练需要的数据，每次执行都会生成一个batch_size的数量的样本（这里进行了样本扩张）

images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)
# 产生训练需要的测试数据，每次执行都会生成一个batch_size的数量的测试样本

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])
# 创建输入数据的placeholder（相当于占位符）

weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
# 第一层权重初始化，产生64个3通道（RGB图片），尺寸为5*5的卷积核，不带L2正则（wl=0.0）
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
# 对输入原始图像进行卷积操作，步长为[1, 1, 1, 1]，即将每一个像素点都计算到，
# 补零模式为'SAME'（不够卷积核大小的块就补充0）
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# 定义第一层的偏置参数，由于有64个卷积核，这里有偏置尺寸为shape=[64]
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
# 卷积结果加偏置后采用relu激活
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
# 第一层的池化操作，使用尺寸为3*3，步长为2*2的池化层进行操作
# 这里的ksize和strides第一个和第四个数字一般都为1
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
# 用LRN对结果进行处理，使得比较大的值变得更大，比较小的值变得更小，模仿神经系统的侧抑制机制

# 这一部分和上面基本相同，不加赘述
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

# 这里定义一个全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
# 将上一层的输出结果拉平（flatten），[batch_size, -1]中的-1代表不确定多大
dim = reshape.get_shape()[1].value
# 得到数据扁平化后的长度

# 建立一个隐含节点数为384的全连接层
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 建立一个隐含节点数为192的全连接层
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 建立输出层（由于cifar数据库一共有10个类别的标签，所以这里输出节点数为10）
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
# 注意这里，这里直接是网络的原始输出（wx+b这种形式），没有加softmax激活

loss = loss(logits, label_holder)
# 计算总体loss，包括weight loss 和 cross_entropy

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 选择AdamOptimizer作为优化器

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
# 关于tf.nn.in_top_k函数的用法见http://blog.csdn.net/uestc_c2_403/article/details/73187915
# tf.nn.in_top_k会返回一个[batch_size, classes(类别数)]大小的布尔型张量，记录是否判断正确

sess = tf.InteractiveSession()  # 注册为默认session
tf.global_variables_initializer().run()  # 初始化全部模型参数

tf.train.start_queue_runners()  # 启动线程（QueueRunner是一个不存在于代码中的东西，而是后台运作的一个概念）

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    # 获得一个batch的训练数据
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch,
                                                          label_holder: label_batch})
    # 运行训练过程并获得一个batch的total_loss

    duration = time.time() - start_time  # 记录跑一个batch所耗费的时间

    if step % 10 == 0:  # 每10个batch输出信息
        examples_per_sec = batch_size / duration  # 计算每秒能跑多少个样本
        sec_per_batch = float(duration)  # 计算每个batch需要耗费的时间

        format_str = (
            'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

# 在测试集上验证精度
num_examples = 10000

num_iter = int(math.ceil(num_examples / batch_size))  # math.ceil 对浮点数向上取整
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    # 获得一个batch的测试数据
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_holder: label_batch})
    true_count += np.sum(predictions)  # 获得预测正确的样本数
    step += 1

precision = true_count / total_sample_count  # 获得预测精度
print("precision = %.4f%%" % (precision * 100))