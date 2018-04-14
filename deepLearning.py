import math
import time
import numpy as np
import tensorflow as tf

max_steps = 500  # 训练轮数（每一轮一个batch参与训练）
isTrain = 1
sess = tf.InteractiveSession()  # 注册为默认session

# 权重初始化函数
# shape：卷积核参数，格式类似于[5,5,3,32]，代表卷积核尺寸（前两个数字），通道数和卷积核个数
# stddev：标准差
# wl：L2正则化的权值参数
# 返回带有L2正则的初始化的权重参数
def variable_with_weight_loss(shape, stddev, wl, nlayer):
    with tf.name_scope('weights'):
        var = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name="w")
        tf.summary.histogram('/weights', var)
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


data_dir = "./fig/TFrecords/traindata.tfrecords-00"  # 数据目录格式

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
batch_size = 30  # 批处理大小
capacity = min_after_dequeue + 3*batch_size  # 批处理容量
images_train, labels_train = tf.train.shuffle_batch([images, labels], batch_size=batch_size,
                                                    capacity=capacity,min_after_dequeue = 10)
# 通过随机打乱的方式创建数据批次

image_holder = tf.placeholder(tf.float32, [batch_size, 60, 160, 1], name = "image")
label_holder = tf.placeholder(tf.int32, [batch_size], name = "label")
# 创建输入数据的placeholder（相当于占位符）

with tf.name_scope('layer1'):
    weight1 = variable_with_weight_loss(shape=[5, 5, 1, 32], stddev=5e-2, wl=0.0, nlayer=1)
    # 第一层权重初始化，产生64个3通道（RGB图片），尺寸为5*5的卷积核，不带L2正则（wl=0.0）
    kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
    # 对输入原始图像进行卷积操作，步长为[1, 1, 1, 1]，即将每一个像素点都计算到，
    # 补零模式为'SAME'（不够卷积核大小的块就补充0）
    with tf.name_scope('biases'):
        bias1 = tf.Variable(tf.constant(0.0, shape=[32]), name = 'b')
        # 定义第一层的偏置参数，由于有64个卷积核，这里有偏置尺寸为shape=[64]
        tf.summary.histogram("layer1" + '/biases', bias1)
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    # 卷积结果加偏置后采用relu激活
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                            padding='SAME')
    # 第一层的池化操作，使用尺寸为3*3，步长为2*2的池化层进行操作
    # 这里的ksize和strides第一个和第四个数字一般都为1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # 用LRN对结果进行处理，使得比较大的值变得更大，比较小的值变得更小，模仿神经系统的侧抑制机制


# 这一部分和上面基本相同，不加赘述
with tf.name_scope('layer2'):
    weight2 = variable_with_weight_loss(shape=[5, 5, 32, 64], stddev=5e-2, wl=0.0, nlayer=2)
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                        padding='SAME')

# 这里定义一个全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
# 将上一层的输出结果拉平（flatten），[batch_size, -1]中的-1代表不确定多大
dim = reshape.get_shape()[1].value
print("dim = ", dim)    #38400
# 得到数据扁平化后的长度

# 建立一个隐含节点数为384的全连接层
with tf.name_scope('layer3'):
    weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004, nlayer=3)
    with tf.name_scope('biases'):
        bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
        tf.summary.histogram('/biases', bias3)
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
    tf.summary.histogram('/outputs', local3)

    # 建立输出层
with tf.name_scope('layer4'):
    weight4 = variable_with_weight_loss(shape=[384, 3], stddev=1/384.0, wl=0.0, nlayer=4)
    bias4 = tf.Variable(tf.constant(0.0, shape=[3]))
    logits = tf.add(tf.matmul(local3, weight4), bias4)
    # 注意这里，这里直接是网络的原始输出（wx+b这种形式），没有加softmax激活

with tf.name_scope('loss'):
    loss = loss(logits, label_holder)
    tf.summary.scalar('loss', loss)
# 计算总体loss，包括weight loss 和 cross_entropy

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
# 选择AdamOptimizer作为优化器

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
# 关于tf.nn.in_top_k函数的用法见http://blog.csdn.net/uestc_c2_403/article/details/73187915
# tf.nn.in_top_k会返回一个[batch_size, classes(类别数)]大小的布尔型张量，记录是否判断正确
keep_prob = tf.placeholder(tf.float32)
# sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

tf.global_variables_initializer().run()  # 初始化全部模型参数

tf.train.start_queue_runners()  # 启动线程（QueueRunner是一个不存在于代码中的东西，而是后台运作的一个概念）
saver = tf.train.Saver()
if isTrain:
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        # 获得一个batch的训练数据
        _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch,
                                                              label_holder: label_batch,keep_prob: 0.5})
        # 运行训练过程并获得一个batch的total_loss

        duration = time.time() - start_time  # 记录跑一个batch所耗费的时间

        if step % 10 == 0:  # 每10个batch输出信息
            examples_per_sec = batch_size / duration  # 计算每秒能跑多少个样本`
            sec_per_batch = float(duration)  # 计算每个batch需要耗费的时间

            format_str = (
                'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
        if step % 5 == 0:
            rs = sess.run(merged, feed_dict={image_holder: image_batch,
                                              label_holder: label_batch})
            writer.add_summary(rs, step)

    #保存模型
    saver.save(sess,"./Model/MyModel", global_step=max_steps)
else:
    ckpt = tf.train.get_checkpoint_state("./Model/")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("load finish!",ckpt.model_checkpoint_path)
    else:
        print("load failed!")
        pass

print("===========before Test========")
# 在测试集上验证精度
num_examples = 600
num_iter = int(math.ceil(num_examples / batch_size))  # math.ceil 对浮点数向上取整
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
classes={"wandao":0, "shizi":0, "zhidao":0}

print("num_iter=",num_iter)
print("total_sample_count",total_sample_count)
while step < num_iter:
    # print(step)
    image_batch, label_batch = sess.run([images_train, labels_train])

    # 获得一个batch的测试数据
    predictions, logits_value = sess.run([top_k_op, logits], feed_dict=
                                                    {image_holder: image_batch,
                                                    label_holder: label_batch})
    true_count += np.sum(predictions)  # 获得预测正确的样本数


    for i in range(batch_size):
        if label_batch[i] ==0 and predictions[i] == True:
            classes["wandao"] += 1
        elif label_batch[i] ==1 and predictions[i] == True:
            classes["shizi"] += 1
        elif label_batch[i] ==2 and predictions[i] == True:
            classes["zhidao"] += 1
    step += 1

print("识别正确数/图像总数: ", true_count, "/", total_sample_count)
precision = true_count / total_sample_count  # 获得预测精度
print("precision = %.4f%%" % (precision * 100))
print(classes)
