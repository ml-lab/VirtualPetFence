
import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from CatFinder import CatFinder

def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

img = load_image('data/cat.jpg')

img = np.tile(img, (32, 1, 1, 1))

catFinder = CatFinder()


segmentedCat = catFinder(img)
print segmentedCat
