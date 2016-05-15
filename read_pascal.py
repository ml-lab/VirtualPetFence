import xml.etree.ElementTree as ET

import operator

import datetime

import os

import scipy.ndimage as ndimage

from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf
import re
import matplotlib.pyplot as plt

import numpy as np

from resnet import _bn, _conv, block, stack

BATCH_SIZE = 32
IMG_SIZE = 224
is_training = tf.convert_to_tensor(True,
                                   dtype='bool',
                                   name='is_training')
true_values = tf.constant(1, dtype=tf.int32, shape=(BATCH_SIZE,))

def weight_variable(shape, scale=1.0, scope='weight'):
    with tf.variable_scope(scope + '_scale_'+str(scale)):
        s = 0.2 * scale / np.sqrt(np.prod(shape[:3]))
        initial = tf.truncated_normal(shape, stddev=s)
        return tf.Variable(initial)


def deconv(x, filters_out, ksize=3, stride=1):
    shape = x.get_shape().as_list()
    filters_in = shape[-1]

    weights = tf.get_variable('weights_transpose', [3, 3, filters_out, filters_in],
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=x.dtype))  # weight_variable(, scale=7.07)

    out_size = [BATCH_SIZE, shape[1] * stride, shape[2] * stride, filters_out]
    return tf.nn.conv2d_transpose(x, weights, out_size, [1, stride, stride, 1])


def upscale(x, scale=2, out_features=None):
    #x =tf.Print(x, [x], message='in upscale', summarize=30)
    with tf.variable_scope('upscale'):
        batch, height, width, in_channels = x.get_shape().as_list()
        if out_features is None:
            out_features = in_channels/2
        W_conv = tf.get_variable('transpose_W', [3, 3, out_features, in_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=x.dtype), dtype=x.dtype)# weight_variable(, scale=7.07)
        x = tf.nn.conv2d_transpose(x, W_conv, [BATCH_SIZE, height*scale, width*scale, out_features], [1, scale, scale, 1])
        #x =tf.Print(x, [x], message='before bn', summarize=30)
        x = _bn(x, out_features, is_training)
        #x =tf.Print(x, [x], message='after bn\n', summarize=30)
        return tf.nn.elu(x)

def gather3D(idx, values):
    N, M, C = values.get_shape().as_list()
    mult = [N*M, M, 1]
    new_ind = tf.reduce_sum(idx*mult, 1)
    return tf.gather(tf.reshape(values, [-1]), new_ind)

def findRegions(x):
    with tf.variable_scope('size56'):
        x = stack(x, 1, 16, bottleneck=True, is_training=is_training, stride=1)
        x = block(x, 16, is_training, 2, bottleneck=True, _conv=deconv)
        #x = upscale(x, out_features=512)
    with tf.variable_scope('size112'):
        x = stack(x, 1, 8, bottleneck=True, is_training=is_training, stride=1)
        x = block(x, 8, is_training, 2, bottleneck=True, _conv=deconv)
        #x = upscale(x, out_features=128)
    with tf.variable_scope('size224'):
        x = stack(x, 1, 4, bottleneck=True, is_training=is_training, stride=1)
        x = block(x, 2, is_training, 1, bottleneck=False)
    return x
    #     x = upscale(x, out_features=64)
    # with tf.variable_scope('size112'):
    #     x = upscale(x, out_features=16)
    # with tf.variable_scope('size224'):
    #     x = upscale(x, out_features=1)
    # return x

def loss(y, batched_indices):
    old_shape = y.get_shape()
    y = tf.reshape(tf.nn.softmax(tf.reshape(y, [BATCH_SIZE, -1])), old_shape[:3])
    y_nonzero = gather3D(batched_indices, y)
    return -tf.reduce_sum(tf.log(y_nonzero))/BATCH_SIZE

def getFilenamesForSegmentedCats():
    filename_paths = open('cats_segmented.txt')
    image_base_path = '/usr/local/data/VOC2012/JPEGImages/'
    segmentations_base_path = '/usr/local/data/VOC2012/SegmentationClass/'
    filenames_images = []
    filenames_segmentations = []

    for line in filename_paths:
        name = line.strip()
        filenames_images.append(image_base_path + name + '.jpg')
        filenames_segmentations.append(segmentations_base_path + name + '.png')
    return filenames_images, filenames_segmentations

def getFilenamesAndMetadata():
    cat_path = 'cats_annotated.txt'
    cat_file = open(cat_path)
    image_names = []
    labels = []
    annotation_files = []
    images_base_path = '/usr/local/data/VOC2012/JPEGImages/'
    annotation_base_path = '/usr/local/data/VOC2012/Annotations/'
    centers = []
    nr_of_points = []
    sizes = []
    for i, line in enumerate(cat_file):
        values = line.split()
        name, l = values[:2]
        size = map(float, values[2:4])

        c = map(float, values[4:][::-1])
        nr_of_points.append(len(c) / 2)
        center = zip([i] * (len(c) / 2),
                     map(lambda x: IMG_SIZE * x / size[0], c[::2]),
                     map(lambda x: IMG_SIZE * x / size[1], c[1::2]))
        centers += sorted(center, key=operator.itemgetter(0, 1, 2))
        annotation_files.append(annotation_base_path + name + '.xml')
        image_names.append(images_base_path + name + '.jpg')
        labels.append(bool(int(l) + 1))
    centers = np.array(centers)
    return image_names, labels, centers, nr_of_points


#tree = ET.parse('/usr/local/data/VOC2012/Annotations/2007_000027.xml')


def getTrainingBatchForSegmentation(image_names, segmentation_names):
    image_names = tf.convert_to_tensor(image_names, dtype=tf.string)
    segmentation_names = tf.convert_to_tensor(segmentation_names, dtype=tf.string)
    image_queue, seg_queue = tf.train.slice_input_producer([image_names, segmentation_names], capacity=BATCH_SIZE*(4+1))
    img = tf.image.decode_jpeg(tf.read_file(image_queue), channels=3)
    img = tf.image.resize_images(img, IMG_SIZE, IMG_SIZE)
    img = img / 255.0

    seg_dtype = tf.float32
    seg = tf.image.decode_png(tf.read_file(seg_queue), channels=3)
    seg = tf.image.resize_images(seg, IMG_SIZE, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    seg = tf.select(tf.logical_and(tf.equal(seg[:, :, 0], 64), tf.equal(tf.reduce_sum(seg, 2), 64)),
                    tf.constant(1, dtype=seg_dtype, shape=(IMG_SIZE, IMG_SIZE)), tf.constant(0, dtype=seg_dtype, shape=(IMG_SIZE, IMG_SIZE)))
    return tf.train.batch([img, tf.expand_dims(seg, 2)], BATCH_SIZE, num_threads=4)


def getTrainingBatch(image_names, labels, nr_of_points):
    centers_range = (np.cumsum([0] + nr_of_points)[:-1]).astype(np.int32)
    nr_of_points = tf.convert_to_tensor(nr_of_points, dtype=tf.int32)
    image_names = tf.convert_to_tensor(image_names, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.bool)
    image_queue, label_queue, roi_idx_queue, nr_of_points_queue = tf.train.slice_input_producer(
        [image_names, labels, centers_range, nr_of_points],
        capacity=BATCH_SIZE * (4 + 1))
    img = tf.image.decode_jpeg(tf.read_file(image_queue), channels=3)
    # img = tf.image.resize_image_with_crop_or_pad(img, 360, 480)
    img = tf.image.resize_images(img, IMG_SIZE, IMG_SIZE)
    img = img / 255.0
    img, label, roi_idx, nr_of_points = tf.train.batch([img, label_queue, roi_idx_queue, nr_of_points_queue], BATCH_SIZE, num_threads=4)
    return img, label, roi_idx, nr_of_points


def showImageAndWeights(img, weights):
    w = weights - weights.min()
    w = (255*(w/weights.max())).astype(np.uint8)
    plt.subplot(1, 2, 1)
    plt.imshow((img*255).astype(np.uint8))
    plt.subplot(1, 2, 2)
    plt.imshow(w)
    plt.show()

def createBatchedIndices(roi_idx, centers, nr_of_points):
    centers = tf.convert_to_tensor(centers, dtype=tf.int32)
    simple_roi_idx = roi_idx

    def inLoop(i, roi_idx):
        new_roi = simple_roi_idx + tf.select(tf.greater(nr_of_points, i), true_values * i,
                                                                      true_values * 0)
        return tf.add(i, 1),\
               tf.concat(0, [roi_idx, new_roi])

    points_max = tf.cast(tf.reduce_max(nr_of_points), tf.int32)

    i = tf.constant(0)
    c = lambda i, nr_of_points: tf.less(i, points_max)
    i2, roi_idx = tf.while_loop(c, inLoop, [i, roi_idx], parallel_iterations=1)

    roi_idx = tf.unique(roi_idx)[0]
    batched_indices = tf.gather(centers, roi_idx)
    unique = tf.unique(batched_indices[:, 0])
    return tf.concat(1, [tf.expand_dims(unique[1], 1), batched_indices[:, 1:]])

def getLastLayersFromGraph(graph):
    scale2 = graph.get_tensor_by_name('scale2/block3/Relu:0') / (10 ** 20)
    scale3 = graph.get_tensor_by_name('scale3/block8/Relu:0') / (10 ** 20)
    scale4 = graph.get_tensor_by_name('scale4/block36/Relu:0') / (10 ** 20)
    scale5 = graph.get_tensor_by_name('scale5/block3/Relu:0') / (10 ** 20)
    return scale2, scale3, scale4, scale5

def segmentationLoss(softmax, seg):
    seg = tf.squeeze(seg, [3])
    return tf.reduce_mean(-seg * tf.log(softmax[:, :, :, 0]) + (seg - 1) * tf.log(softmax[:, :, :, 1]))

def stichMutipleLayers(graph):
    with tf.variable_scope('Joining_layers'):
        scale2, scale3, scale4, scale5 = getLastLayersFromGraph(graph)
        with tf.variable_scope('up2'):
            up2 = block(scale2, 4, is_training, stride=1, bottleneck=False)
        with tf.variable_scope('up3'):
            up3 = block(scale3, 4, is_training, stride=2, bottleneck=False, _conv=deconv)
        with tf.variable_scope('up4'):
            up4 = block(scale4, 4, is_training, stride=4, bottleneck=False, _conv=deconv)
        with tf.variable_scope('up5'):
            with tf.variable_scope('a'):
                up5 = block(scale5, 4, is_training, stride=2, bottleneck=False, _conv=deconv)
            with tf.variable_scope('b'):
                up5 = block(up5, 4, is_training, stride=4, bottleneck=False, _conv=deconv)
        return tf.concat(3, [up2, up3, up4, up5])# tf.concat(3, [up2, up3, up4, up5])


def restore_resnet_graph(sess):
    layers = 152
    new_saver = tf.train.import_meta_graph(meta_fn(layers))
    new_saver.restore(sess, checkpoint_fn(layers))

# image_names, labels, centers, nr_of_points = getFilenamesAndMetadata()
# img, label, roi_idx, nr_of_points = getTrainingBatch(image_names, labels, nr_of_points)
# batched_indices = createBatchedIndices(roi_idx, centers, nr_of_points)
#
# print "graph restored"
#
sess = tf.Session()



restore_resnet_graph(sess)
graph = tf.get_default_graph()
#prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")
#tf.image_summary('images', images)
# last_layers = list(getLastLayersFromGraph(graph))
# #with tf.device('/gpu:0'):
#
stiched = stichMutipleLayers(graph)
pred_region = findRegions(stiched)
old_shape = pred_region.get_shape()
softmaxed = tf.reshape(tf.nn.softmax(tf.reshape(pred_region, (BATCH_SIZE*IMG_SIZE*IMG_SIZE, 2))), old_shape)
#pred_max = tf.reduce_max(pred_region)
tf.histogram_summary('pred_region', pred_region)
tf.image_summary('predicted regions', tf.expand_dims(softmaxed[:, :, :, 0], 3))
# cross_entropy = loss(pred_region, batched_indices)
# train_step = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(cross_entropy)#, var_list=tf.trainable_variables()[-18:])

img_names, seg_names = getFilenamesForSegmentedCats()
img, seg = getTrainingBatchForSegmentation(img_names, seg_names)
tf.image_summary('image', img)
tf.image_summary('segmentation', seg)

cross_entropy = segmentationLoss(softmaxed, seg)
tf.scalar_summary('cross_entropy', cross_entropy)
train_step = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(cross_entropy)#, var_list=tf.trainable_variables()[-18:])

saver = tf.train.Saver(tf.all_variables())
summary_op = tf.merge_all_summaries()

init = tf.initialize_all_variables()
sess.run(init)

train_dir = '/usr/local/models/catnet'

step = 0
load_old = True
if tf.gfile.Exists(train_dir) and load_old:
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(train_dir,
                                         ckpt.model_checkpoint_path))

    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    step_str = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('Succesfully loaded model from %s at step=%s.' %
          (ckpt.model_checkpoint_path, step_str))
    step = int(step_str)

tf.train.start_queue_runners(sess=sess)
summary_writer = tf.train.SummaryWriter(train_dir,
                                        graph=sess.graph)


def saveAndSummary(saver, summary_op, step, sess):
    summary_str = sess.run(summary_op, feed_dict={images: i})
    summary_writer.add_summary(summary_str, step)
    saver.save(sess, '/usr/local/models/catnet/model.ckpt', global_step=step)


for step in range(step, 100000):
    i, l = sess.run([img, seg])
    loss_val, _ = sess.run([cross_entropy, train_step], feed_dict={images: i})
    #s = sess.run(last_layers, feed_dict={images: i})
    #showImageAndWeights(i, s)
    #print sess.run(prob_tensor, feed_dict={images: i})
    #prob = sess.run(prob_tensor, feed_dict={images: i})
    #pred = np.argsort(prob, axis=1)[:, -5:]
    #pred = np.argmax(prob, axis=1)
    # pred_bool_bot = pred > 280
    # pred_bool_top = pred < 294
    # pred_bool = np.logical_and(pred_bool_bot, pred_bool_top).sum(1).astype(np.bool)
    # acc = np.mean((pred_bool == l).astype(np.float))
    #top1 = print_prob(prob[0])
    print step, ':', loss_val
    if step % 100 == 0:
        saveAndSummary(saver, summary_op, step, sess)



print l
plt.imshow(i)
