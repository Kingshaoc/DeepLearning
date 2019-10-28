import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

# 手写数字识别
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
# 拆分为验证集和训练集 0-5000  5000-60000
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


def show_single_image(img_arr):
    plt.imshow(img_arr, cmap='binary')
    plt.show()



show_single_image(x_train[0])  # 28*28

class_names=['t-shirt','trouser','pullover','dress','coat','scandal','shirt','sneaker','bag','ankleboot']
def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize = (n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap="binary",
                       interpolation = 'nearest')
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot']
show_imgs(3, 5, x_train, y_train, class_names)

"""
#tf.keras.models.Sequential()
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))#输入是28*28的图像 用faltten展平
model.add(keras.layers.Dense(300,activation='relu'))#全连接层 relu=max(0,x)
model.add(keras.layers.Dense(100,activation='relu'))#全连接层
#10分类的问题 softmax:将向量变成概率分布 y=[e^x1/sum,e^x2/sum,e^x3/sum]
model.add(keras.layers.Dense(10,activation='softmax'))
"""


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# reason for sparse: y->index. y->one_hot->[]
# sgd
# 还关心准确率
model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])

model.layers
#dense=235500
# [None, 784] * W + b -> [None, 300] W.shape [784, 300], b = [300]
model.summary() #模型的概况

history=model.fit(x_train,y_train,epochs=10,validation_data=(x_valid, y_valid))

print(type(history))

print(history.history)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

model.evaluate(x_test, y_test)
