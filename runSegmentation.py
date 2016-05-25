import cv2
import skimage.io
import skimage.transform

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from CatFinder import CatFinder

from CatPlayer import CatPlayer
from DrawGui import DrawArea


def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

#img = load_image('/usr/local/data/ludde/DSC_0065.jpg')

#img = np.tile(img, (32, 1, 1, 1))
alpha = 0.6

draw = DrawArea()
drawing = draw.doDraw()


def nothing(x):
    global alpha
    alpha = x/100.

catPlayer = CatPlayer()


cv2.namedWindow('cam')
cv2.createTrackbar('alpha','cam',50,100,nothing)
cap = cv2.VideoCapture(0)
rval, frame = cap.read()


catFinder = CatFinder('/tmp/models/catnet12')
xrange = np.arange(224)
xv, yv = np.meshgrid(xrange, xrange)

up_scale_y = 640/224.
up_scale_x = 480/224.

cat_list = range(100)

cnt = 0
def saveImages(img_list, name):
    for i in range(len(img_list)): cv2.imwrite("cat_" + str(i) + ".jpg", img_list[i])
while rval:
    print 'alpha:', alpha
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32)/255
    img = img[np.newaxis, :, :, ::-1]
    frame[drawing] = (0, 0, 255)

    segmentedCat = np.squeeze(catFinder.getProbability(img))# > alpha
    segmentedCatBig = cv2.resize(segmentedCat, (640, 480)) > alpha
    frame[segmentedCatBig] = 0.5*frame[segmentedCatBig] + frame[segmentedCatBig]*np.array([0.5, 0, 0])
    if segmentedCat.max() > 0.90:
        print 'IS CAT'
        prod_dist = segmentedCat/segmentedCat.sum()
        centerx = np.sum(prod_dist*xv)*up_scale_x
        centery = np.sum(prod_dist*yv)*up_scale_y
        cv2.circle(frame, (int(centery), int(centerx)), 4, (0, 255, 0))
        if drawing[centerx, centery]:
            catPlayer.play()
        else:
            catPlayer.pause()

    #FOR VIDEO
    cat_list[cnt%100] = frame
    cnt += 1
    cv2.imshow('cam', frame)
    rval, frame = cap.read()
    key = cv2.waitKey(1)
    if key == 27:
        break


segmentedCat = catFinder(img)


seg = np.zeros((224, 224, 3))
seg[:, :, 2] = segmentedCat
plt.imshow(0.5*seg + 0.5*img[0])
print segmentedCat

