import operator

import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = 224

true_values = tf.constant(1, dtype=tf.int32, shape=(BATCH_SIZE,))

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


def getProducerForFilenames(image_names, segmentation_names):
    image_names = tf.convert_to_tensor(image_names, dtype=tf.string)
    segmentation_names = tf.convert_to_tensor(segmentation_names, dtype=tf.string)
    return tf.train.slice_input_producer([image_names, segmentation_names], capacity=BATCH_SIZE*(4+1))

def getTrainingBatchForSegmentation(producer, SEED = 1, distort=True):
    image_queue, seg_queue = producer
    img = tf.image.decode_jpeg(tf.read_file(image_queue), channels=3)
    seg = tf.image.decode_png(tf.read_file(seg_queue), channels=3)

    if distort:
        new_size = tf.random_uniform([1], minval=int(1.25*IMG_SIZE), maxval=int(IMG_SIZE*1.8), dtype=tf.int32)
        new_aspect = tf.random_uniform([1], minval=0.8, maxval=1.2)
        new_width = tf.cast(tf.cast(new_size, tf.float32)*new_aspect, tf.int32)

        img_seg = tf.pack([img, seg])
        img_seg = tf.image.resize_nearest_neighbor(img_seg, tf.pack([new_size[0], new_width[0]]))
        img_seg = tf.random_crop(img_seg, [2, IMG_SIZE, IMG_SIZE, 3])
        img, seg = tf.unpack(img_seg)
    else:
        img = tf.image.resize_images(img, IMG_SIZE, IMG_SIZE)
        seg = tf.image.resize_images(seg, IMG_SIZE, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    img = tf.cast(img, tf.float32)
    if distort:
        img = tf.image.random_flip_left_right(img, seed=SEED)
        img = tf.image.random_brightness(img, max_delta=63)
        img = tf.image.random_contrast(img,
                                 lower=0.2, upper=1.8)
    img = img - tf.reduce_min(img)
    img = img / tf.reduce_max(img)

    seg_dtype = tf.float32
    if distort:
        seg = tf.image.random_flip_left_right(seg, seed=SEED)
    seg = tf.select(tf.logical_and(tf.equal(seg[:, :, 0], 64), tf.equal(tf.reduce_sum(seg, 2), 64)),
                    tf.constant(1, dtype=seg_dtype, shape=(IMG_SIZE, IMG_SIZE)), tf.constant(0, dtype=seg_dtype, shape=(IMG_SIZE, IMG_SIZE)))
    return tf.train.batch([img, tf.expand_dims(seg, 2)], BATCH_SIZE, num_threads=4, capacity=32*2)


def getDataset(filename='cats_segmented_mixed.txt', scope='retrieve_training_data', distort=True):
    with tf.variable_scope(scope):
        img_names, seg_names = getFilenamesForSegmentedCats(filename)
        return getTrainingBatchForSegmentation(getProducerForFilenames(img_names, seg_names), distort=distort)


def getTrainingBatchForCachedData(filename='/usr/local/data/segimages.bin'):
    # filename_queue = tf.train.string_input_producer(['/usr/local/data/segimages.bin'])

    image_bytes = IMG_SIZE*IMG_SIZE*3
    segmentation_bytes = IMG_SIZE*IMG_SIZE
    s2_bytes = 56*56*256
    s3_bytes = 28*28*512
    s4_bytes = 14*14*1024
    s5_bytes = 7*7*2048

    record_bytes = image_bytes + segmentation_bytes + s2_bytes+s3_bytes+s4_bytes+s5_bytes
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