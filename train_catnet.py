import os

import numpy as np
import tensorflow as tf

import resnet
from read_data import getDataset, getTestBatchWithNoSegmentation, getFilenamesForSegmentedCats, \
    getTrainingBatchForSegmentation, getProducerForFilenames
from resnet import _bn, block, stack

UPDATE_OPERATORS = 'update_operators'

BATCH_SIZE = 32
IMG_SIZE = 224

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
    shape = tf.shape(x)
    shape_list = x.get_shape().as_list()
    filters_in = shape_list[-1]

    weights = tf.get_variable('weights_transpose', [3, 3, filters_out, filters_in],
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=x.dtype))  # weight_variable(, scale=7.07)
    out_size = tf.pack([shape[0], shape[1] * stride, shape[2] * stride, filters_out])
    upscaled = tf.nn.conv2d_transpose(x, weights, out_size, [1, stride, stride, 1])
    upscaled.set_shape([shape_list[0], shape_list[1] * stride, shape_list[2] * stride, filters_out])
    return upscaled


def findRegions(x):
    with tf.variable_scope('size56'):
        x = stack(x, 1, 16, bottleneck=False, is_training=is_training, stride=1)
        x = block(x, 16, is_training, 2, bottleneck=False, _conv=deconv)
    with tf.variable_scope('size112'):
        x = stack(x, 1, 8, bottleneck=False, is_training=is_training, stride=1)
        x = block(x, 8, is_training, 2, bottleneck=False, _conv=deconv)
    with tf.variable_scope('size224'):
        x = stack(x, 1, 4, bottleneck=False, is_training=is_training, stride=1)
        x = block(x, 2, is_training, 1, bottleneck=False)
    return x

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
    graph = tf.get_default_graph()
    return graph, graph.get_tensor_by_name("images:0")


def accuracy(softmax, label):
    y = tf.cast(tf.expand_dims(tf.argmax(softmax, 3), 3), tf.float32)
    return 1 - tf.reduce_mean(tf.abs(label - (1 - y)))


def makeSummary(summary_op, step, sess):
    summary_str = sess.run(summary_op, feed_dict={is_training: True, images: i, segmented_label: l})
    summary_writer.add_summary(summary_str, step)

    i_val, l_val = sess.run([img_val, seg_val])

    loss_val2, acc_val, test_summary_str = sess.run([cross_entropy, acc, summary_op],
                                  feed_dict={images: i_val, segmented_label: l_val, is_training: False})
    test_writer.add_summary(test_summary_str, step)
    print 'Validate loss:', loss_val2, 'Validate acc', acc_val


def findAvgVal(sess, img_val, seg_val):
    from sklearn.metrics import average_precision_score
    avg_acc = 0
    sm_avg = 0
    for i in range(1, 101):
        i_val, l_val = sess.run([img_val, seg_val])

        sm, acc_val = sess.run([softmaxed, acc],
                                                    feed_dict={images: i_val, segmented_label: l_val,
                                                               is_training: False})
        sm_cat = sm[:, :, :, 0]
        precision = average_precision_score(l_val.ravel(), sm_cat.ravel())
        sm_avg += precision
        avg_acc += acc_val

        print acc_val,  avg_acc/i, precision, sm_avg
    print "AVG", avg_acc/100, sm_avg/100


def loadModelFromFile(load_dir):
    step = 1
    if tf.gfile.Exists(load_dir) and load_old:
        restorer = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if os.path.isabs(ckpt.model_checkpoint_path):
            restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
            restorer.restore(sess, os.path.join(load_dir,
                                                ckpt.model_checkpoint_path))

        step_str = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, step_str))
        if train_dir == load_dir:
            step = int(step_str) + 1
        return step


def inference(graph):
    #Stich togheter different layers
    stiched = stichMutipleLayers(getLastLayersFromGraph(graph))

    #Predict and upscale
    pred_region = findRegions(stiched)
    old_shape = pred_region.get_shape().as_list()
    softmaxed = tf.reshape(tf.nn.softmax(tf.reshape(pred_region, (-1, 2))), [-1] + old_shape[1:])
    tf.histogram_summary('pred_region', pred_region)
    tf.image_summary('predicted regions', tf.expand_dims(softmaxed[:, :, :, 0], 3))
    return softmaxed


def getAllFilenames():
    global img_coco_names, seg_coco_names, img_coco_names2, seg_coco_names2, img_default_names, seg_default_names
    img_coco_names, seg_coco_names = getFilenamesForSegmentedCats('/usr/local/data/coco2014_cats/segmented_cats.txt',
                                                                  image_base_path='/usr/local/data/coco2014_cats/images/',
                                                                  segmentation_base_path='/usr/local/data/coco2014_cats/segmentations/')
    img_coco_names2, seg_coco_names2 = getFilenamesForSegmentedCats('/usr/local/data/coco2014_cats2/segmented_cats.txt',
                                                                    image_base_path='/usr/local/data/coco2014_cats2/images/',
                                                                    segmentation_base_path='/usr/local/data/coco2014_cats2/segmentations/')
    img_default_names, seg_default_names = getFilenamesForSegmentedCats('data/cats_segmented_mixed.txt')


    return (img_coco_names + img_coco_names2 + img_default_names), (seg_coco_names + seg_coco_names2 + seg_default_names)

if __name__=='__main__':
    load_old = True
    train_dir = '/tmp/models/catnet12'
    test_dir = '/tmp/models/catnet12_test'
    load_dir = train_dir#'/tmp/models/catnet12'

    #Setup data-collection graph
    img_filenames, seg_filenames = getAllFilenames()
    img, seg = getTrainingBatchForSegmentation(getProducerForFilenames(img_filenames, seg_filenames), distort=True)
    img_val, seg_val = getDataset('data/cats_segmented_val.txt', scope='retrieve_test_data', distort=False)

    #Load pretrained
    sess = tf.Session()
    graph,images = restore_resnet_graph(sess)
    tf.image_summary('images', images)

    #placeholder for labels
    segmented_label = tf.placeholder(tf.float32, (None, IMG_SIZE, IMG_SIZE, 1), 'segmented_label')

    softmaxed = inference(graph)

    cross_entropy = segmentationLoss(softmaxed, segmented_label)
    tf.scalar_summary('cross_entropy', cross_entropy)
    acc = accuracy(softmaxed, segmented_label)
    tf.scalar_summary('acc', acc)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(0.9)
    apply_avg_loss =  ema.apply([cross_entropy])
    loss_avg = ema.average(cross_entropy)
    tf.scalar_summary('loss_avg', loss_avg)

    optimizer = tf.train.AdamOptimizer(0.0001)
    grads = optimizer.compute_gradients(cross_entropy)
    apply_gradients = optimizer.apply_gradients(grads)

    batchnorm_updates = tf.get_collection(resnet.UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_step = tf.group(apply_gradients, batchnorm_updates_op, apply_avg_loss)

    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)


    saver = tf.train.Saver(tf.all_variables(), max_to_keep=10, keep_checkpoint_every_n_hours=2)
    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    sess.run(init)

    if load_old:
        step = loadModelFromFile(load_dir)
    else:
        step = 1

    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.train.SummaryWriter(train_dir,
                                            graph=sess.graph)
    test_writer = tf.train.SummaryWriter(test_dir,
                                            graph=sess.graph)

    avg_loss = 0
    cnt = 1
    for step in range(step, 100000):
        i, l = sess.run([img, seg])
        loss_val, _ = sess.run([cross_entropy, train_step], feed_dict={images: i, segmented_label: l, is_training: True})

        avg_loss += loss_val
        print step, ':', loss_val, '\t', avg_loss/cnt
        cnt += 1
        if step % 100 == 0:
            cnt = 1
            print 'Avg loss:', avg_loss / 100
            avg_loss = 0
            makeSummary(summary_op, step, sess)
        if step % 1000 == 0:
            saver.save(sess, train_dir + '/model.ckpt', global_step=step)
