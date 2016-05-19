import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
annType = ['segm','bbox']
annType = annType[0]      #specify type here
print 'Running demo for *%s* results.'%(annType)

#initialize COCO ground truth api
dataDir='/usr/local/data/coco2014'
dataType='train2014'
annFile = '%s/annotations/instances_%s.json'%(dataDir,dataType)
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile='%s/results/instances_%s_fake%s100_results.json'
resFile = resFile%(dataDir, dataType, annType)

# visialuze gt and dt side by side
catIds = cocoGt.getCatIds(catNms=[ 'cat'])
imgIds=sorted(cocoGt.getImgIds(catIds=catIds))
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]
img = cocoGt.loadImgs(imgId)[0]
I = io.imread('%s/train2014/%s'%(dataDir,img['file_name']))

# visialuze gt and dt side by side
fig = plt.figure(figsize=[15,10])

# ground truth
plt.subplot(121)
plt.imshow(I); plt.axis('off'); plt.title('ground truth')
annIds = cocoGt.getAnnIds(imgIds=imgId)
anns = cocoGt.loadAnns(annIds)
cocoGt.showAnns(anns)