import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np

BATCH_SIZE = 32
IMG_SIZE = 224

def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers

def meta_fn(layers):
    return 'ResNet-L%d.meta' % layers

def restore_resnet_graph(sess):
    layers = 152
    new_saver = tf.train.import_meta_graph(meta_fn(layers))
    new_saver.restore(sess, checkpoint_fn(layers))

def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

def getFilenamesForSegmentedCats(filename='cats_segmented_mixed.txt'):
    filename_paths = open(filename)
    image_base_path = '/usr/local/data/VOC2012/JPEGImages/'
    segmentations_base_path = '/usr/local/data/VOC2012/SegmentationClass/'
    filenames_images = []
    filenames_segmentations = []

    for line in filename_paths:
        name = line.strip()
        filenames_images.append(image_base_path + name + '.jpg')
        filenames_segmentations.append(segmentations_base_path + name + '.png')
    return filenames_images, filenames_segmentations

img = load_image('data/cat.jpg')

sess = tf.Session()
restore_resnet_graph(sess)
graph = tf.get_default_graph()
images = graph.get_tensor_by_name("images:0")

scale2 = graph.get_tensor_by_name('scale2/block3/Relu:0') / (10 ** 20)
scale3 = graph.get_tensor_by_name('scale3/block8/Relu:0') / (10 ** 20)
scale4 = graph.get_tensor_by_name('scale4/block36/Relu:0') / (10 ** 20)
scale5 = graph.get_tensor_by_name('scale5/block3/Relu:0') / (10 ** 20)

img_names, seg_names = getFilenamesForSegmentedCats('cats_segmented_val.txt')
out_array = np.array([], dtype=np.float32)
for (im_n, seg_n) in zip(img_names, seg_names):
    print im_n, seg_n
    img = load_image(im_n).astype(np.float32)
    seg = load_image(seg_n).astype(np.float32)
    seg = np.logical_and(seg[:, :, 0] == 64, seg.sum(2) == 64).astype(np.float32)
    s = sess.run([scale2, scale3, scale4, scale5], feed_dict={images: img.reshape((1, IMG_SIZE, IMG_SIZE, 3))})
    for arr in [img, seg, s[0], s[1], s[2], s[3]]:
        out_array = np.append(out_array, arr)
out_array.tofile('/usr/local/data/segimages_val.bin')

print scale2, scale3, scale4, scale5