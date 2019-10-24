# Interceptionnet 实现思想 分组卷积
#网易云 吴恩达链接
#https://mooc.study.163.com/learn/2001281004?tid=2001392030&_trace_c_p_k2_=b9867e4ad84c406096463d1eed0c3270#/learn/content?type=detail&id=2001729331
#Inception v3是一个性价比叫搞的选择 3*3 =1*3+3*1
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

#定义interception分组卷积块
# x:输入
#output_channel_for_each_path：每组的输出通道数 [10,20,5]
#name
def  inception_block(x,output_channel_for_each_path,name):
    #scope的定义是为了防止命名冲突
    with tf.variable_scope(name):
        #1*1的卷积层
        conv1_1=tf.layers.conv2d(x,output_channel_for_each_path[0],(1,1),strides=(1,1),padding='same',activation=tf.nn.relu,name='conv1')
        #3*3的卷积层
        conv3_3 = tf.layers.conv2d(x, output_channel_for_each_path[1], (3, 3), strides=(1, 1), padding='same',
                                   activation=tf.nn.relu, name='conv3_3')
        #5*5的卷积层
        conv5_5 = tf.layers.conv2d(x, output_channel_for_each_path[2], (5, 5), strides=(1, 1), padding='same',
                                   activation=tf.nn.relu, name='conv5_5')
        #maxpooling的分支
        max_pooling=tf.layers.max_pooling2d(x,(2,2),(2,2),name='max_pooling')
        #对max_pooling增加padding
        max_pooling_shape=max_pooling.get_shape().as_list()[1:]
        input_shape=x.get_shape().as_list()[1:]
        width_padding=(input_shape[0]-max_pooling_shape[0])//2
        height_padding=(input_shape[1]-max_pooling_shape[0])//2
        padded_pooling=tf.pad(max_pooling,[[0,0],[width_padding,width_padding],[height_padding,height_padding],[0,0]])
        #拼接结果 axis=3表示在第四个通道上做拼接
        concat_layer=tf.concat([conv1_1,conv3_3,conv5_5,padded_pooling],axis=3)
        return concat_layer


# 卷积所需的输入矩阵 是图片 32 *32 *3 将每行的数据转为原始的图片
x_image = tf.reshape(x, [-1, 3, 32, 32])  # 32*32*3
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])  # 交换通道


#interceptionNet:卷积-池化-卷积-池化-interceptionBblock-interceptionBblock-interceptionBblock-maxpool-fulliconnection-dropout-softmax

conv1 = tf.layers.conv2d(x_image, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv1_1')

pooling1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), name='pool1')

# inceptionNet
inception_2a=inception_block(pooling1,[16,16,16],name="inception_2a")
inception_2b=inception_block(inception_2a,[16,16,16],name="inception_2b")
#池化
pooling2 = tf.layers.max_pooling2d(inception_2b, (2, 2), (2, 2), name='pool2')


# inceptionNet
inception_3a=inception_block(pooling2,[16,16,16],name="inception_3a")
inception_3b=inception_block(inception_3a,[16,16,16],name="inception_3b")
#池化
pooling3 = tf.layers.max_pooling2d(inception_3b, (2, 2), (2, 2), name='pool3')

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
