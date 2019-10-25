import numpy as np

vgg16_data = np.load('vgg16.npy', encoding='latin1',allow_pickle=True)
print(type(vgg16_data)) #ndarray


data_dict=vgg16_data.item()
print(data_dict.keys())#conv5_1 fc6 conv5_3
print(len(data_dict))

conv1_1=data_dict['conv1_1']
print(len(conv1_1))#(3,3,3,64)
w,b=conv1_1
print(w.shape)
print(b.shape)


fc6 = data_dict['fc6']
print(len(fc6))

w, b = fc6
print(w.shape)
print(b.shape)