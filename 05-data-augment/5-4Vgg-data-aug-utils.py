#tensorflow数据增强的api
#1.resize
#2.crop
#3.flip
#4.brigthness & contrast

import  numpy as np
import  tensorflow as tf
import  matplotlib.pyplot as plt
from matplotlib.pyplot import  imshow

name='./gugong.jpg'
img_string=tf.read_file(name) #将图片以字符串的形式读入
img_decoded=tf.image.decode_image(img_string)

sess=tf.Session()
#(365,600,3)
img_decoded_val=sess.run(img_decoded)
print(img_decoded_val.shape)

imshow(img_decoded_val)
plt.show()

#resize
#tf.image.resize_area :
#tf.image.resize_bicubic 二次线性插值法 缩小的时候没有损失 放大的时候像素值用二次线性函数进行计算
# tf.image.resize_nearest_neighbor 方法的过程中用最相近的像素点的值
name='./gugong.jpg'
img_string=tf.read_file(name) #将图片以字符串的形式读入
img_decoded=tf.image.decode_image(img_string)
img_decoded=tf.reshape(img_decoded,[1,365,600,3])

resize_img=tf.image.resize_bicubic(img_decoded,[730,1200])

sess=tf.Session()
#(365,600,3)
img_decoded_val=sess.run(resize_img)
img_decoded_val=img_decoded_val.reshape((730,1200,3))
img_decoded_val = np.asarray(img_decoded_val, np.uint8)# float-int
print(img_decoded_val.shape)

imshow(img_decoded_val)
plt.show()

# crop 裁剪
# tf.image.pad_to_bounding_box 把图像做到一个画布上
# tf.image.crop_to_bounding_box
# tf.random_crop 随机裁剪
name = './gugong.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded, [1, 365, 600, 3])

#50：在新画布的位置 100：在新画布的位置 500：画布的大小 800：画布的大小
padded_img = tf.image.pad_to_bounding_box(
    img_decoded, 50, 100, 500, 800)

sess = tf.Session()
img_decoded_val = sess.run(padded_img)
img_decoded_val = img_decoded_val.reshape((500, 800, 3))
img_decoded_val = np.asarray(img_decoded_val, np.uint8)
print(img_decoded_val.shape)

imshow(img_decoded_val)
plt.show()


# tf.image.flip_up_down
# tf.image.flip_left_right
# tf.image.random_flip_up_down
# tf.image.random_flip_left_right

name = './gugong.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded, [1, 365, 600, 3])

flipped_img = tf.image.flip_up_down(img_decoded)

sess = tf.Session()
img_decoded_val = sess.run(flipped_img)
img_decoded_val = img_decoded_val.reshape((365, 600, 3))
img_decoded_val = np.asarray(img_decoded_val, np.uint8)
print(img_decoded_val.shape)
imshow(img_decoded_val)
plt.show()


# brightness
# tf.image.adjust_brightness
# tf.image.random_brightness
# tf.image.adjust_constrast
# tf.image.random_constrast
name = './gugong.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded, [1, 365, 600, 3])

new_img = tf.image.adjust_brightness(img_decoded, 0.5)

sess = tf.Session()
img_decoded_val = sess.run(new_img)
img_decoded_val = img_decoded_val.reshape((365, 600, 3))
img_decoded_val = np.asarray(img_decoded_val, np.uint8)
print(img_decoded_val.shape)
imshow(img_decoded_val)
plt.show()