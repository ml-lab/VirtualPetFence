import os
import tensorflow as tf
import numpy as np



class CatFinder:

    def __init__(self):
        self.sess = tf.Session()
        train_dir = '/tmp/models/catnet12'

        ckpt = tf.train.get_checkpoint_state(train_dir)
        model_checkpoint_path = ckpt.model_checkpoint_path.split('/')[-1]
        if not os.path.isabs(model_checkpoint_path):
            model_checkpoint_path = os.path.join(train_dir, model_checkpoint_path)
        saver = tf.train.import_meta_graph(model_checkpoint_path + '.meta')
        saver.restore(self.sess, model_checkpoint_path)
        step_str = model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s' % (model_checkpoint_path, step_str))

        self.graph = tf.get_default_graph()
        self.images = self.graph.get_tensor_by_name('images:0')
        self.is_training = self.graph.get_tensor_by_name('is_training:0')
        self.softmaxed = self.graph.get_tensor_by_name('Reshape_1:0')
        self.out_label = tf.argmax(self.softmaxed, 3)
        self.prob_tensor = self.graph.get_tensor_by_name("prob:0")

    def __call__(self, img):
        return 1- self.sess.run(self.out_label, feed_dict={self.images: img, self.is_training: False})

    def isCat(self, img):
        top10 = np.argsort(self.sess.run(self.prob_tensor, feed_dict={self.images: img, self.is_training: False}))[:5]
        return np.logical_and(top10 > 281, top10 < 287).sum() > 0

    def getProbability(self, img):
        return self.sess.run(self.softmaxed, feed_dict={self.images: img, self.is_training: False})[:, :, :, 0]