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

# 权值的维度和x的维度是一样的 3072*1
w = tf.get_variable('w', [x.get_shape()[-1], 10],
                    initializer=tf.random_normal_initializer(0, 1))

# 偏置 1
b = tf.get_variable('b', [10],
                    initializer=tf.constant_initializer(0.0))

# 输出[None,3072]*[3072*10]=[None,10]
y_ = tf.matmul(x, w) + b

# e^x/sum(e^x)   [[0.01,0.9,...0.03],[0.01,0.02,...0.7]]
p_y = tf.nn.softmax(y_)
# onehot编码将y变成一个分部 5->[0,0,0,0,1,0,0,0,0,0]
y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)

# 平凡差损失函数
#loss = tf.reduce_mean(tf.square(y_one_hot - p_y))

#交叉熵损失函数
# y_->softmax
# y->onehot
# loss=ylogy_
loss=tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)

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
