import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import resnet
from read_data import getDataset
from resnet import _bn, block, stack

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
    with tf.variable_scope('upscale'):
        batch, height, width, in_channels = x.get_shape().as_list()
        if out_features is None:
            out_features = in_channels/2
        W_conv = tf.get_variable('transpose_W', [3, 3, out_features, in_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=x.dtype), dtype=x.dtype)# weight_variable(, scale=7.07)
        x = tf.nn.conv2d_transpose(x, W_conv, [BATCH_SIZE, height*scale, width*scale, out_features], [1, scale, scale, 1])
        x = _bn(x, is_training)
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

def loss(y, batched_indices):
    old_shape = y.get_shape()
    y = tf.reshape(tf.nn.softmax(tf.reshape(y, [BATCH_SIZE, -1])), old_shape[:3])
    y_nonzero = gather3D(batched_indices, y)
    return -tf.reduce_sum(tf.log(y_nonzero))/BATCH_SIZE


def getLastLayersFromGraph(graph):
    scale2 = graph.get_tensor_by_name('scale2/block3/Relu:0') / (10)
    scale3 = graph.get_tensor_by_name('scale3/block8/Relu:0') / (10 ** 5)
    scale4 = graph.get_tensor_by_name('scale4/block36/Relu:0') / (10 ** 20)
    scale5 = graph.get_tensor_by_name('scale5/block3/Relu:0') / (10 ** 30)
    return scale2, scale3, scale4, scale5

def segmentationLoss(softmax, seg):
    tf.image_summary('segmentation', seg)
    seg = tf.squeeze(seg, [3])
    return tf.reduce_mean(-seg * tf.log(softmax[:, :, :, 0]) + (seg - 1) * tf.log(softmax[:, :, :, 1]))

def depth_conv(x, filters_out):
    with tf.variable_scope('depth_conv'):
        x_conv = resnet._conv(x, filters_out, ksize=3)
        x_conv = _bn(x_conv, is_training)
        return resnet._relu(x_conv)


def increaseToSpatialSize(x, out_filters=None, size=56):
    B, N, M, C = x.get_shape().as_list()
    with tf.variable_scope('size' + str(N)):
        if out_filters is None:
            out_filters = C
        if N>=size: return block(x, out_filters, is_training, stride=1, bottleneck=False)
        return increaseToSpatialSize(block(x, C, is_training, 2, bottleneck=False, _conv=deconv))

def stichMutipleLayers(scales):
    with tf.variable_scope('Joining_layers'):
        scale2, scale3, scale4, scale5 = scales
        with tf.variable_scope('up2'):
            scale2 = depth_conv(scale2, 32)
            up2 = block(scale2, 8, is_training, stride=1, bottleneck=False)
        with tf.variable_scope('up3'):
            scale3 = depth_conv(scale3, 32)
            up3 = block(scale3, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
        with tf.variable_scope('up4'):
            with tf.variable_scope('a'):
                scale4 = depth_conv(scale4, 32)
                up4 = block(scale4, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
            with tf.variable_scope('b'):
                up4 = block(up4, 8, is_training, stride=2, bottleneck=False, _conv=deconv)
        with tf.variable_scope('up5'):
            with tf.variable_scope('a'):
                scale5 = depth_conv(scale5, 32)
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




sess = tf.Session()
restore_resnet_graph(sess)
graph = tf.get_default_graph()
#sess.run(tf.initialize_all_variables())

# with tf.Graph().as_default() as my_graph:
#     img, seg = getDataset()
#     prob_tensor, =tf.import_graph_def(graph.as_graph_def(), input_map={"images:0": img}, return_elements=["prob:0"])
#     sess2 = tf.Session(graph=my_graph)
#     sess2.run(tf.initialize_all_variables())
#     print sess2.run(prob_tensor)

prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")

stiched = stichMutipleLayers(getLastLayersFromGraph(graph))



pred_region = findRegions(stiched)
old_shape = pred_region.get_shape()
softmaxed = tf.reshape(tf.nn.softmax(tf.reshape(pred_region, (BATCH_SIZE*IMG_SIZE*IMG_SIZE, 2))), old_shape)

tf.histogram_summary('pred_region', pred_region)
tf.image_summary('predicted regions', tf.expand_dims(softmaxed[:, :, :, 0], 3))

img, seg = getDataset('cats_segmented_mixed.txt', distort=True)
img_val, seg_val = getDataset('cats_segmented_val.txt', scope='retrieve_test_data')

segmented_label = tf.placeholder(tf.float32, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), 'segmented_label')
#segmented_label = loaded[0]
tf.image_summary('images', images)

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

train_dir = '/tmp/models/catnet8'
test_dir = '/tmp/models/catnet8_test'

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
        total_loss = 0
        cnt = 1
        print 'Avg loss:', avg_loss / 100
        avg_loss = 0
        saveAndSummary(summary_op, step, sess)
    if step % 1000 == 0:
        saver.save(sess, train_dir + '/model.ckpt', global_step=step)

print l
plt.imshow(i)
