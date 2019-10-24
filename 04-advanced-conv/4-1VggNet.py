# Vggnet  3*3的卷积或者是1*1的卷积 加深层次 6个卷积层 每两个卷积层后面加了一个池化
# https://blog.csdn.net/qq_33037903/article/details/88622679【经典卷积神经网络结构——AlexNet网络结构详解（卷积神经网络入门，Keras代码实现）】
# https://www.jianshu.com/p/d6f556c1ff47
# https://my.oschina.net/u/876354/blog/1634322 ****
# vgg net 加深版的alexnet
# 1、LRN层无性能增益（A-LRN）
# VGG作者通过网络A-LRN发现，AlexNet曾经用到的LRN层（local response normalization，局部响应归一化）并没有带来性能的提升，因此在其它组的网络中均没再出现LRN层。
# 2、随着深度增加，分类性能逐渐提高（A、B、C、D、E）
# 从11层的A到19层的E，网络深度增加对top1和top5的错误率下降很明显。
# 3、多个小卷积核比单个大卷积核性能好（B）
# VGG作者做了实验用B和自己一个不在实验组里的较浅网络比较，较浅网络用conv5x5来代替B的两个conv3x3，结果显示多个小卷积核比单个大卷积核效果要好。
# 1、通过增加深度能有效地提升性能；
# 2、最佳模型：VGG16，从头到尾只有3x3卷积与2x2池化，简洁优美；
# 3、卷积可代替全连接，可适应各种尺寸的图片
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

# 卷积所需的输入矩阵 是图片 32 *32 *3 将每行的数据转为原始的图片
x_image = tf.reshape(x, [-1, 3, 32, 32])  # 32*32*3
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])  # 交换通道

# 卷积层
# 32 ouput channel number
# 卷积核 kernel size
# padding same 输出的图片大小不变  32*32*32（通道）
conv1_1 = tf.layers.conv2d(x_image, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv1_1')
# vggnet添加新的卷积层
conv1_2 = tf.layers.conv2d(conv1_1, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv1_2')

# 池化 kernel size(2,2)，步长(2,2) 16*16*32（通道）
pooling1 = tf.layers.max_pooling2d(conv1_2, (2, 2), (2, 2), name='pool1')

# 第二个卷积层
conv2_1 = tf.layers.conv2d(pooling1, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv2_1')
# vgg 添加新的卷积层
conv2_2 = tf.layers.conv2d(conv2_1, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv2_2')

# 池化 kernel size(2,2)，步长(2,2)  8*8
pooling2 = tf.layers.max_pooling2d(conv2_2, (2, 2), (2, 2), name='pool2')

# 第三个卷积层
conv3_1 = tf.layers.conv2d(pooling2, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv3_1')
# vgg 添加新的卷积层
conv3_2 = tf.layers.conv2d(conv3_1, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv3_2')

# 池化 kernel size(2,2)，步长(2,2) 4*4
pooling3 = tf.layers.max_pooling2d(conv3_2, (2, 2), (2, 2), name='pool3')

# 全连接层 4*4*32
# [None,4*4*32]
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
train_steps = 10000  # 70.45%
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
