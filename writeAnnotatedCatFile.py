import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt
import itertools
import numpy as np



#tree = ET.parse('/usr/local/data/VOC2012/Annotations/2007_000027.xml')
cat_path = '/usr/local/data/VOC2012/ImageSets/Main/cat_train.txt'
cat_file = open(cat_path)
image_names = []
labels = []
names = []
annotation_files = []
images_base_path = '/usr/local/data/VOC2012/JPEGImages/'
annotation_base_path = '/usr/local/data/VOC2012/Annotations/'

for line in cat_file:
    name, l = line.split()
    names.append(name)
    annotation_files.append(annotation_base_path + name + '.xml')
    image_names.append(images_base_path + name + '.jpg')
    labels.append(l)

def getBoxforObject(object):
    return map(lambda x: int(x.text), o.find('bndbox'))

def findCenter(xmin, xmax):
    return xmax - (xmax - xmin)/2.0

out = open('cats_annotated.txt', 'w')
for i, a in enumerate(annotation_files):
    root = ET.parse(a).getroot()
    objects = root.findall('object')
    size = root.find('size')
    height = int(size[1].text)
    width = int(size[0].text)
    centers = []
    for o in objects:
        xmin, ymin, xmax, ymax = getBoxforObject(o)
        centers.append((findCenter(xmin, xmax), findCenter(ymin, ymax)))
    if len(centers):
        out.write(names[i] +" "+ labels[i] + " " + str(height) + " " + str(width) + " " + " ".join(map(str, itertools.chain.from_iterable(centers))) + "\n")
out.close()