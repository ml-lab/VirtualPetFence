import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sess = tf.Session()

def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img
img = load_image('data/cat.jpg')

with tf.device('/cpu:0'):

    x = tf.placeholder(tf.float32, name='x')
    flippin = tf.image.random_flip_left_right(image=x, seed=2)
    flippin2 = tf.image.random_flip_left_right(image=x, seed=1)

    for i in range(100):
        f1, f2 = sess.run([flippin, flippin2], feed_dict={x: img})
        print np.equal(f1, f2).mean()
#a = np.fromfile('/usr/local/data/segimages.bin', dtype=np.float32)
#plt.imshow(a[:224*224*3].reshape((224,224,3)))
#print 'hei9'