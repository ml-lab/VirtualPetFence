import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def getCatSegmentationDataset():
    image_names_path = 'train_segment.txt'
    images_base_path = '/usr/local/data/VOC2012/SegmentationClass/'
    out_file = open('cats_segmented.txt', 'w')
    imagenames_file = open(image_names_path)
    image_names = []
    for line in imagenames_file:
        name = line.strip()
        seg_image = ndimage.imread(images_base_path + name + '.png')
        N, M, C = seg_image.shape
        isCat = np.sum(np.logical_and(seg_image[:, :, 0] == 64, np.sum(seg_image, 2) == 64 )) > 0
        if isCat > 0:
            out_file.write(line)


getCatSegmentationDataset()