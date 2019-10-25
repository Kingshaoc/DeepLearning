#
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
        self._data = self._data
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
batch_size=20
x = tf.placeholder(tf.float32, [batch_size, 3072])
# (None) 1维
y = tf.placeholder(tf.int64, [batch_size])

# 卷积所需的输入矩阵 是图片 32 *32 *3 将每行的数据转为原始的图片
x_image = tf.reshape(x, [-1, 3, 32, 32])  # 32*32*3
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])  # 交换通道
#将所有的样本切分为一个一个的样本 在每个样本上做数据增强
x_image_arr=tf.split(x_image,num_or_size_splits=batch_size,axis=0) #在第0维度进行切分
#设置一个数组保存每个图像进行数据增强之后的结果
result_x_image_arr = []

for x_single_image in x_image_arr:
    #x_single_image [1,32,32,3]
    x_single_image = tf.reshape(x_single_image, [32, 32, 3])
    data_aug_1=tf.image.random_flip_left_right(x_single_image)
    data_aug_2=tf.image.random_brightness(data_aug_1,max_delta=63)
    data_aug_3=tf.image.random_contrast(data_aug_2,lower=0.2,upper=1.8)
    x_single_image = tf.reshape(data_aug_3, [1, 32, 32, 3])
    result_x_image_arr.append(x_single_image)

result_x_images = tf.concat(result_x_image_arr, axis=0)#将所有增强后的图片拼接起来，在第0维度进行合并
normal_result_x_images = result_x_images / 127.5 - 1


# 卷积层
# 32 ouput channel number
# 卷积核 kernel size
# padding same 输出的图片大小不变  32*32*32（通道）
conv1_1 = tf.layers.conv2d(normal_result_x_images, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv1_1')
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

#给变量的统计量建立summary
def variable_summary(var,name):
    with tf.name_scope(name):
        mean=tf.reduce_mean(var)#均值
        # 平方差
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('mean',mean)
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('min',tf.reduce_mean(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        #直方图
        tf.summary.histogram('histogram',var)

with tf.name_scope('summary'):
    variable_summary(conv1_1, 'conv1_1')
    variable_summary(conv1_2, 'conv1_2')
    variable_summary(conv2_1, 'conv2_1')
    variable_summary(conv2_2, 'conv2_2')
    variable_summary(conv3_1, 'conv3_1')
    variable_summary(conv3_2, 'conv3_2')



# tensorboard实现可视化的方法 所有和Tensorboard有关的方法都在tf.summary里面
# 'loss':<10,1,1> <20,1.08>
loss_summary = tf.summary.scalar('loss', loss)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)


# 归一化的逆过程

inputs_summary = tf.summary.image('inputs_image', x_image)

# merge loss accuracy input 训练过程  merge所有的summary
merged_summary = tf.summary.merge_all()
# 第二种方法  test过程
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])

# 指定输出到具体的文件夹
LOG_DIR = '.'
run_label = 'run_vgg_tensorboard'
run_dir = os.path.join(LOG_DIR, run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)
train_log_dir = os.path.join(run_dir, 'train')  # train数据文件夹
test_log_dir = os.path.join(run_dir, 'test')  # test数据文件夹
# 判断文件夹是否存在
if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)

# 初始化
init = tf.global_variables_initializer()

train_steps = 10000
test_steps = 100
out_put_summary_every_steps = 100  # 每一百次去计算outputsummary
# 78.04% -82.6%（图像增强之后）
with tf.Session() as sess:
    sess.run(init)
    # 输出到具体的文件夹中
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)  # 指定计算图
    test_writer = tf.summary.FileWriter(test_log_dir)  # 不指定计算图
    fixed_test_batch_data, fixed_test_batch_labels \
        = test_data.next_batch(batch_size)

    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        eval_ops = [loss, accuracy, train_op]
        should_output_summary = ((i + 1) % out_put_summary_every_steps == 0)
        if should_output_summary:
            eval_ops.append(merged_summary)
        eval_ops_result = sess.run(eval_ops, feed_dict={x: batch_data, y: batch_labels})
        loss_val,acc_val=eval_ops_result[0:2] #前两个不会变
        if should_output_summary:
            train_summary_str=eval_ops_result[-1] #merged_summary
            train_writer.add_summary(train_summary_str,i+1)#指定第几步输出的
            test_summary_str = sess.run([merged_summary_test],
                                        feed_dict={
                                            x: fixed_test_batch_data,
                                            y: fixed_test_batch_labels,
                                        })[0]
            test_writer.add_summary(train_summary_str,i+1)
        if (i + 1) % 100 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i + 1, loss_val, acc_val))
        if (i + 1) % 1000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy], feed_dict={x: test_batch_data, y: test_batch_labels})
                all_test_acc_val.append(test_acc_val)

            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i + 1, test_acc))
