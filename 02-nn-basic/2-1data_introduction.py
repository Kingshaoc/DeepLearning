import pickle
import numpy as np
import os
import matplotlib.pyplot as  plt

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

with open(os.path.join(CIFAR_DIR, "data_batch_1"), 'rb') as f:
    data = pickle.load(f, encoding='bytes')
    print(type(data))
    print(data.keys())
    print(type(data[b'data']))  # ndarray
    print(type(data[b'labels']))  # list
    print(type(data[b'batch_label']))  # bytes
    print(type(data[b'filenames']))  # list
    print(data[b'data'].shape)  # 1万张图片 10000*3072  3073=32*32*3
    print(data[b'data'][0:2])
    print(len(data[b'labels']))  # 10000个图片每个图片所对应的标签
    print(data[b'labels'][0:2])

    # 反推图片
    image_arr = data[b'data'][100]  # 取第一百行的数据
    image_arr = image_arr.reshape((3, 32, 32))  # 32 32 3
    image_arr = image_arr.transpose((1, 2, 0))  # 通道的交换 32*32*3

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow

    imshow(image_arr)
