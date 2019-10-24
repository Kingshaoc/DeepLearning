# MobileNet：深度可分离卷积
# https://zhuanlan.zhihu.com/p/31551004
# 为什么要设计mobilenet？ https://www.jianshu.com/p/854cb5857070
#3*3conv （省去了batchNormalization步骤） ->rule-1x1 conv-relu
#将输入的每个通道都拆开 在每个通道上做卷积

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

# 搭建tensorflow placeholder占位符 None*3072列 None代表输入的样本数量是不确定的
x = tf.placeholder(tf.float32, [None, 3072])
# (None) 1维
y = tf.placeholder(tf.int64, [None])

#深度可分离卷积
# x:输入
#out_channel_num：经过1*1conv后的输出的通道数目
#name
def  separable_conv_block(x,out_channel_num,name):
    #scope的定义是为了防止命名冲突
    with tf.variable_scope(name):
        input_channel=x.get_shape().as_list()[-1]#-1表示最后一列 或得输入的通道数
         #channel_wise_x是一个列表[channel1,channel2,...] tensorflow api split
        channel_wise_x = tf.split(x, input_channel, axis = 3)
        output_channels=[]
        for i in range(len(channel_wise_x)):
            #注意这里只有一个通道 分到极致 3*3*1（1个通道）*很多个split 最后1个通道1个通道concat起来
            output_channel=tf.layers.conv2d(channel_wise_x[i],1,(3,3),strides=(1,1),padding='same',activation=tf.nn.relu, name = 'conv_%d' % i)
            output_channels.append(output_channel)
        #拼接起来 concat
        concat_layer=tf.concat(output_channels,axis=3)
        #经过1*1的卷积
        conv1_1=tf.layers.conv2d(concat_layer,out_channel_num,(1,1),strides=(1,1),padding='same',activation=tf.nn.relu,name='conv1_1')
        return conv1_1


# 卷积所需的输入矩阵 是图片 32 *32 *3 将每行的数据转为原始的图片
x_image = tf.reshape(x, [-1, 3, 32, 32])  # 32*32*3
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])  # 交换通道


conv1 = tf.layers.conv2d(x_image, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv1_1')

pooling1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), name='pool1')

# separable_Net
separable_2a=separable_conv_block(pooling1,32,name="separable_2a")
separable_2b=separable_conv_block(separable_2a,32,name="separable_2b")

#池化
pooling2 = tf.layers.max_pooling2d(separable_2b, (2, 2), (2, 2), name='pool2')


# separable_Net
separable_3a=separable_conv_block(pooling2,32,name="separable_3a")
separable_3b=separable_conv_block(separable_3a,32,name="separable_3b")
#池化
pooling3 = tf.layers.max_pooling2d(separable_3b, (2, 2), (2, 2), name='pool3')

# 全连接层

flatten = tf.layers.flatten(pooling3)

y_ = tf.layers.dense(flatten, 10)

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
train_steps = 10000
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
