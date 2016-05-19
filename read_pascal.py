import xml.etree.ElementTree as ET

import operator

import datetime

import os

import scipy.ndimage as ndimage

import tensorflow as tf
import re
import matplotlib.pyplot as plt

import numpy as np

import resnet
from resnet import _bn, _conv, block, stack

UPDATE_OPERATORS = 'update_operators'

BATCH_SIZE = 32
IMG_SIZE = 224

true_values = tf.constant(1, dtype=tf.int32, shape=(BATCH_SIZE,))
is_training = tf.placeholder(dtype=tf.bool,
                          name='is_training')


def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers

def meta_fn(layers):
    return 'ResNet-L%d.meta' % layers

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
        x = _bn(x, is_training)
        #x =tf.Print(x, [x], message='after bn\n', summarize=30)
        return tf.nn.elu(x)

def gather3D(idx, values):
    N, M, C = values.get_shape().as_list()
    mult = [N*M, M, 1]
    new_ind = tf.reduce_sum(idx*mult, 1)
    return tf.gather(tf.reshape(values, [-1]), new_ind)

def findRegions(x):
    with tf.variable_scope('size56'):
        x = stack(x, 1, 16, bottleneck=False, is_training=is_training, stride=1)
        x = block(x, 16, is_training, 2, bottleneck=False, _conv=deconv)
        #x = upscale(x, out_features=512)
    with tf.variable_scope('size112'):
        x = stack(x, 1, 8, bottleneck=False, is_training=is_training, stride=1)
        x = block(x, 8, is_training, 2, bottleneck=False, _conv=deconv)
        #x = upscale(x, out_features=128)
    with tf.variable_scope('size224'):
        x = stack(x, 1, 4, bottleneck=False, is_training=is_training, stride=1)
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

def getFilenamesForSegmentedCats(filename='cats_segmented.txt'):
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

def getFilenamesAndMetadata(cat_path='cats_annotated.txt'):
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
    SEED = 1
    image_names = tf.convert_to_tensor(image_names, dtype=tf.string)
    segmentation_names = tf.convert_to_tensor(segmentation_names, dtype=tf.string)
    image_queue, seg_queue = tf.train.slice_input_producer([image_names, segmentation_names], capacity=BATCH_SIZE*(4+1))
    img = tf.image.decode_jpeg(tf.read_file(image_queue), channels=3)
    img = tf.image.resize_images(img, IMG_SIZE, IMG_SIZE)
    img = tf.image.random_flip_left_right(img, seed=SEED)

    img = tf.image.random_brightness(img, max_delta=63)
    img = tf.image.random_contrast(img,
                             lower=0.2, upper=1.8)
    img = img - tf.reduce_min(img)
    img = img / tf.reduce_max(img)

    seg_dtype = tf.float32
    seg = tf.image.decode_png(tf.read_file(seg_queue), channels=3)
    seg = tf.image.random_flip_left_right(seg, seed=SEED)
    seg = tf.image.resize_images(seg, IMG_SIZE, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    seg = tf.select(tf.logical_and(tf.equal(seg[:, :, 0], 64), tf.equal(tf.reduce_sum(seg, 2), 64)),
                    tf.constant(1, dtype=seg_dtype, shape=(IMG_SIZE, IMG_SIZE)), tf.constant(0, dtype=seg_dtype, shape=(IMG_SIZE, IMG_SIZE)))
    return tf.train.batch([img, tf.expand_dims(seg, 2)], BATCH_SIZE, num_threads=4, capacity=32*2)


def getDataset(filename='cats_segmented_mixed.txt', scope='retrieve_training_data'):
    with tf.variable_scope(scope):
        img_names, seg_names = getFilenamesForSegmentedCats(filename)
        return getTrainingBatchForSegmentation(img_names, seg_names)

def getTrainingBatchForCachedData(filename='/usr/local/data/segimages.bin'):
    # filename_queue = tf.train.string_input_producer(['/usr/local/data/segimages.bin'])

    image_bytes = IMG_SIZE*IMG_SIZE*3
    segmentation_bytes = IMG_SIZE*IMG_SIZE
    s2_bytes = 56*56*256
    s3_bytes = 28*28*512
    s4_bytes = 14*14*1024
    s5_bytes = 7*7*2048

    record_bytes = image_bytes + segmentation_bytes + s2_bytes+s3_bytes+s4_bytes+s5_bytes
    # reader = tf.FixedLengthRecordReader(record_bytes=record_bytes*4)
    # key, value = reader.read(filename_queue)
    # record_float = tf.decode_raw(value, tf.float32)
    # img = tf.reshape(tf.slice(record_float, [0], [image_bytes]), (IMG_SIZE, IMG_SIZE, 3))
    # seg = tf.reshape(tf.slice(record_float, [image_bytes], [segmentation_bytes]), (IMG_SIZE, IMG_SIZE, 1))
    # s2 = tf.reshape(record_float[segmentation_bytes:segmentation_bytes+s2_bytes], (56, 56, 256))
    # s3_slice = record_float[s2_bytes:s2_bytes+s3_bytes]# tf.slice(record_float, [s2_bytes], [s3_bytes])
    # s3 = tf.reshape(s3_slice, (28, 28, 512))
    # s4 = tf.reshape(record_float[s3_bytes:s3_bytes+s4_bytes], (14, 14, 1024))
    # s5 = tf.reshape(record_float[s4_bytes: s4_bytes + s5_bytes], (7, 7, 2048))
    # min_capacity = 100
    with tf.device('/cpu:0'):
        in_data = tf.constant(np.fromfile(filename, dtype=np.float32).reshape((-1, record_bytes)), name='in_data')
        record_float = tf.train.slice_input_producer([in_data], capacity=96)[0]
        img = tf.reshape(tf.slice(record_float, [0], [image_bytes]), (IMG_SIZE, IMG_SIZE, 3))
        seg = tf.reshape(tf.slice(record_float, [image_bytes], [segmentation_bytes]), (IMG_SIZE, IMG_SIZE, 1))
        s2 = tf.reshape(record_float[segmentation_bytes:segmentation_bytes+s2_bytes], (56, 56, 256))
        s3_slice = record_float[s2_bytes:s2_bytes+s3_bytes]# tf.slice(record_float, [s2_bytes], [s3_bytes])
        s3 = tf.reshape(s3_slice, (28, 28, 512))
        s4 = tf.reshape(record_float[s3_bytes:s3_bytes+s4_bytes], (14, 14, 1024))
        s5 = tf.reshape(record_float[s4_bytes: s4_bytes + s5_bytes], (7, 7, 2048))
        return tf.train.batch([seg, s2, s3, s4, s5], BATCH_SIZE, num_threads=4, capacity=32 * 2)
    #return tf.train.shuffle_batch([img, seg], batch_size=BATCH_SIZE, min_after_dequeue=100, capacity=100+3*BATCH_SIZE, num_threads=1)
    #return tf.train.shuffle_batch([seg, s2, s3, s4, s5], batch_size=BATCH_SIZE, min_after_dequeue=BATCH_SIZE*4, capacity=50000+3*BATCH_SIZE, num_threads=4)

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

def stichMutipleLayers(scales):
    with tf.variable_scope('Joining_layers'):
        scale2, scale3, scale4, scale5 = scales
        with tf.variable_scope('up2'):
            up2 = block(scale2, 8, is_training, stride=1, bottleneck=False)
        with tf.variable_scope('up3'):
            up3 = block(scale3, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
        with tf.variable_scope('up4'):
            with tf.variable_scope('a'):
                up4 = block(scale4, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
            with tf.variable_scope('b'):
                up4 = block(up4, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
        with tf.variable_scope('up5'):
            with tf.variable_scope('a'):
                up5 = block(scale5, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
            with tf.variable_scope('b'):
                up5 = block(up5, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
            with tf.variable_scope('c'):
                up5 = block(up5, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
        return tf.concat(3, [up2, up3, up4, up5])# tf.concat(3, [up2, up3, up4, up5])


def restore_resnet_graph(sess):
    layers = 152
    new_saver = tf.train.import_meta_graph(meta_fn(layers))
    new_saver.restore(sess, checkpoint_fn(layers))


def accuracy(softmax, label):
    y = tf.cast(tf.expand_dims(tf.argmax(softmax, 3), 3), tf.float32)
    return 1 - tf.reduce_mean(tf.abs(label - (1 - y)))

# image_names, labels, centers, nr_of_points = getFilenamesAndMetadata()
# img, label, roi_idx, nr_of_points = getTrainingBatch(image_names, labels, nr_of_points)
# batched_indices = createBatchedIndices(roi_idx, centers, nr_of_points)
#
# print "graph restored"
#


sess = tf.Session()



restore_resnet_graph(sess)
graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")


stiched = stichMutipleLayers(getLastLayersFromGraph(graph))


#is_eval = tf.placeholder(dtype=tf.bool, name='is_eval')
#loaded = tf.cond(is_eval, lambda: getTrainingBatchForCachedData('/usr/local/data/segimages_val.bin'), lambda: getTrainingBatchForCachedData())
#tf.train.start_queue_runners(sess=sess)
#bla = sess.run(loaded)
#print bla
#stiched = stichMutipleLayers(loaded[1:])



pred_region = findRegions(stiched)
old_shape = pred_region.get_shape()
softmaxed = tf.reshape(tf.nn.softmax(tf.reshape(pred_region, (BATCH_SIZE*IMG_SIZE*IMG_SIZE, 2))), old_shape)

tf.histogram_summary('pred_region', pred_region)
tf.image_summary('predicted regions', tf.expand_dims(softmaxed[:, :, :, 0], 3))

img, seg = getDataset()
img_val, seg_val = getDataset('cats_segmented_val.txt', scope='retrieve_test_data')

segmented_label = tf.placeholder(tf.float32, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), 'segmented_label')
#segmented_label = loaded[0]
tf.image_summary('images', images)
tf.image_summary('segmentation', segmented_label)

cross_entropy = segmentationLoss(softmaxed, segmented_label)
tf.scalar_summary('cross_entropy', cross_entropy)
acc = accuracy(softmaxed, segmented_label)
tf.scalar_summary('acc', acc)

# loss_avg
ema = tf.train.ExponentialMovingAverage(0.9)
apply_avg_loss =  ema.apply([cross_entropy])
loss_avg = ema.average(cross_entropy)
tf.scalar_summary('loss_avg', loss_avg)
#train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cross_entropy)#, var_list=tf.trainable_variables()[-18:])

optimizer = tf.train.AdamOptimizer(0.0001)#.minimize(cross_entropy)#, var_list=tf.trainable_variables()[-18:])
grads = optimizer.compute_gradients(cross_entropy)
apply_gradients = optimizer.apply_gradients(grads)

batchnorm_updates = tf.get_collection(resnet.UPDATE_OPS_COLLECTION)
batchnorm_updates_op = tf.group(*batchnorm_updates)
train_step = tf.group(apply_gradients, batchnorm_updates_op, apply_avg_loss)

for grad, var in grads:
    if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)


saver = tf.train.Saver(tf.all_variables(), max_to_keep=10, keep_checkpoint_every_n_hours=2)
#variables_without_batch_norm = filter(lambda x: x.name.find('moving_')==-1, tf.all_variables())
summary_op = tf.merge_all_summaries()

init = tf.initialize_all_variables()
sess.run(init)

train_dir = '/tmp/models/catnet5'
test_dir = '/tmp/models/catnet5_test'

step = 1
load_old = True
if tf.gfile.Exists(train_dir) and load_old:
    restorer = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        restorer.restore(sess, ckpt.model_checkpoint_path)
    else:
        # Restores from checkpoint with relative path.
        restorer.restore(sess, os.path.join(train_dir,
                                         ckpt.model_checkpoint_path))

    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    step_str = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('Succesfully loaded model from %s at step=%s.' %
          (ckpt.model_checkpoint_path, step_str))
    step = int(step_str) + 1

tf.train.start_queue_runners(sess=sess)
summary_writer = tf.train.SummaryWriter(train_dir,
                                        graph=sess.graph)
test_writer = tf.train.SummaryWriter(test_dir,
                                        graph=sess.graph)

def saveAndSummary(summary_op, step, sess):
    summary_str = sess.run(summary_op, feed_dict={is_training: True, images: i, segmented_label: l})
    summary_writer.add_summary(summary_str, step)

    i_val, l_val = sess.run([img_val, seg_val])

    loss_val2, acc_val, test_summary_str = sess.run([cross_entropy, acc, summary_op],
                                  feed_dict={images: i_val, segmented_label: l_val, is_training: False})
    test_writer.add_summary(test_summary_str, step)
    print 'Validate loss:', loss_val2, 'Validate acc', acc_val


total_loss = 0
avg_loss = 0
cnt = 1
loss_validation = cross_entropy*10
tf.scalar_summary('validation_loss', loss_validation)
for step in range(step, 100000):
    i, l = sess.run([img, seg])
    loss_val, _ = sess.run([cross_entropy, train_step], feed_dict={images: i, segmented_label: l, is_training: True})

    avg_loss += loss_val
    total_loss += loss_val
    print step, ':', loss_val, '\t', total_loss/cnt
    cnt += 1
    if step % 100 == 0:
        print 'Avg loss:', avg_loss / 100
        avg_loss = 0
        saveAndSummary(summary_op, step, sess)
    if step % 1000 == 0:
        saver.save(sess, train_dir + '/model.ckpt', global_step=step)

print l
plt.imshow(i)
