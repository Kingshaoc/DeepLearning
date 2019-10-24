# ResNet 卷积 池化 +多个残差块（降采样）
# 残差连接结构 输入分为两个部分：1通过卷积层做一些事情2.恒等变化降采样（maxpolling） 最后两个输出相加
# https://blog.csdn.net/weixin_43624538/article/details/85049699
import tensorflow as tf
import os
import pickle
import numpy as np

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


# 加载数据的函数 从plickle中读取数据
# 获取图片的像素值以及标签值
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


# tensorflow dataset 数据加载
# cifar10数据输入的处理函数
class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)  # 10个类别
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)  # 成为一个矩阵
        self._data = self._data / 127.5 - 1  # 0-255的数做归一化
        self._labels = np.hstack(all_labels)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] ->[5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    # 返回batch_size个样本作为一个batch
    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator:end_indicator]
        batch_lables = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_lables


train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

# (10000,3072)
train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)


# 残差连接块生成函数 x:输入 output_channel：输出的通道数
def residual_block(x, output_channel):
    input_channel = x.get_shape().as_list()[-1]  # 输入的通道数
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2, 2)  # 需要降采样
    elif input_channel == output_channel:  # 恒等变化
        increase_dim = False  # 不增加通道数
        strides = (1, 1)
    else:
        raise Exception("input channel cant match output channel")

    conv1 = tf.layers.conv2d(x, output_channel, (3, 3), strides=strides, padding='same', activation=tf.nn.relu,
                             name='conv1')  # 降采样步长为2 32*32*3->16*16*6
    conv2 = tf.layers.conv2d(conv1, output_channel, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu,
                             name='conv2')  # 不需要将采样了16*16*6-》16*16*6

    if increase_dim:  # 如果需要增加通道的数目的话，就需要降采样 1：average_pooling2d
        # [None,witdth,height,channel] ->[,,,channel*2]
        pooled_x = tf.layers.average_pooling2d(x, (2, 2), (2, 2), padding='valid')  # pooling不会增加通道数 x:32*32*3 ->16*16*3
        #
        padded_x = tf.pad(pooled_x,
                          [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])  # 增加通道数 16*16*3->16*16*6
    else:
        padded_x = x;
    # f(x):残差+x：恒等变化  卷积+恒等变换
    # 两种情况：1. 通道数目变成两倍2.  不做通道变化
    output_x = conv2 + padded_x
    return output_x


# num_residual_blocks[3,4,6,3](论文)[2,3,2] 每一层的残差连接块  num_filter_base:最初的通道数目 class_num：适用多种数据集
def res_net(x, num_residual_blocks, num_filter_base, class_num):
    num_subsampling = len(num_residual_blocks)  # 3
    layers = []
    # x：[None,witdth,height,channel]->[width,height,channel]
    input_size = x.get_shape().as_list()[1:]
    with tf.variable_scope('conv0'):  # 第一个的卷积层
        conv0 = tf.layers.conv2d(x, num_filter_base, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu,
                                 name='conv0')
        layers.append(conv0)
    for sample_id in range(num_subsampling):  # 3
        for i in range(num_residual_blocks[sample_id]):  # [2,3,2] 第一次调用残差块构造函数构造两个残差块  第二次构造3个残差块 第二次构造2个残差块
            with tf.variable_scope("conv%d_%d" % (sample_id, i)):
                # x:layers[-1]取上一层残差块的构造结果作为输入
                # out_put channel:输出的通道数  num_filter_base:最初的通道数目*2^(sample_id) 第一层残差块 3*2^0   第二层残差块 3*2^1    3*2^2
                conv = residual_block(layers[-1], num_filter_base * (2 ** sample_id))
            layers.append(conv)

    multiplier = 2 ** (num_subsampling - 1)  # 2的平方 32*32->8*8 通道数3->12
    assert layers[-1].get_shape().as_list()[1:] \
           == [input_size[0] / multiplier,
               input_size[1] / multiplier,
               num_filter_base * multiplier]
    # average pooling fc 最后的全连接层：
    with tf.variable_scope("fc"):
        # 最终的池化层
        # layers[-1]：[None,witdth,height,channel] 在1，2两个维度上做pooling
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        # class_num：10个分类
        logits = tf.layers.dense(global_pool, class_num)
        layers.append(logits)

    # 返回最后一层 作为输出
    return layers[-1]


# 搭建tensorflow placeholder占位符 None*3072列 None代表输入的样本数量是不确定的
x = tf.placeholder(tf.float32, [None, 3072])
# (None) 1维
y = tf.placeholder(tf.int64, [None])

# 卷积所需的输入矩阵 是图片 32 *32 *3 将每行的数据转为原始的图片
x_image = tf.reshape(x, [-1, 3, 32, 32])  # 32*32*3
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])  # 交换通道

# [2,3,2]每一层的残差块 一个残差块有两层conv 第一层2个残差块 ，第二层3个残差块 第三层2个残差块 一共7*2+1+1 =16层
# 32最初的通道数
# 没经过一个残差层维度降为原来的一半 通道数增加一倍
y_ = res_net(x_image, [2, 3, 2], 32, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# bool
predict = tf.arg_max(y_, 1)

# [1,0,1,1,1,0]
correct_prediction = tf.equal(predict, y)

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# 梯度下降
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 初始化
init = tf.global_variables_initializer()
batch_size = 20
train_steps = 1000
test_steps = 100
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        [loss_val, acc_val, _] = sess.run([loss, accuracy, train_op], feed_dict={x: batch_data, y: batch_labels})
        if (i + 1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i + 1, loss_val, acc_val))
        if (i + 1) % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy], feed_dict={x: test_batch_data, y: test_batch_labels})
                all_test_acc_val.append(test_acc_val)

            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i + 1, test_acc))
