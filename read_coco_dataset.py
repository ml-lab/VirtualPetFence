import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from skimage.draw import polygon
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
annType = ['segm','bbox']
annType = annType[0]      #specify type here
print 'Running demo for *%s* results.'%(annType)

#initialize COCO ground truth api
dataDir='/usr/local/data/coco2014'
dataType='val2014'
writeDir='/usr/local/data/coco2014_cats2'

annFile = '%s/annotations/instances_%s.json'%(dataDir,dataType)
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile='%s/results/instances_%s_fake%s100_results.json'
resFile = resFile%(dataDir, dataType, annType)


writeImages = True
# visialuze gt and dt side by side
filenames = open(writeDir+'/segmented_cats.txt', 'w')
catIds = cocoGt.getCatIds(catNms=[ 'cat'])
imgIds=sorted(cocoGt.getImgIds(catIds=catIds))
imgs = cocoGt.loadImgs(imgIds)
imgs = {i['id']: i for i in imgs}
annIds = cocoGt.getAnnIds(imgIds=imgIds, catIds=[17])
anns = cocoGt.loadAnns(annIds)
anns_dict = {}
for ann in anns:
    if ann['image_id'] not in anns_dict:
        anns_dict[ann['image_id']] = ann
    else:
        anns_dict[ann['image_id']]['segmentation'] += ann['segmentation']
anns = anns_dict.values()
for ann in anns:
    if ann['segmentation'] == 'counts' or 'counts' in ann['segmentation']: continue
    img = imgs[ann['image_id']]
    assert(img['id'] == ann['image_id'])
    filenames.write(img['file_name'].replace('.jpg', '\n'))
    if not writeImages: continue

    I = io.imread('%s/val2014/%s'%(dataDir,img['file_name']))
    if len(I.shape) < 3:
        I = np.tile(I[:, :, np.newaxis], (1, 1, 3))

    mask = np.zeros(I.shape, dtype=np.uint8)
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((len(seg)/2, 2))
        rr, cc = polygon(poly[:, 1], poly[:, 0])
        mask[rr, cc] = [64, 0, 0]

    io.imsave(writeDir + '/images/' + img['file_name'], I)
    io.imsave(writeDir + '/segmentations/' + img['file_name'].replace('jpg', 'png'), mask)
    print 'Wrote file: ', writeDir + '/' + img['file_name']

filenames.close()

imgId = imgIds[np.random.randint(100)]
img = cocoGt.loadImgs(imgId)[0]
I = io.imread('%s/train2014/%s'%(dataDir,img['file_name']))

# visialuze gt and dt side by side
fig = plt.figure(figsize=[15,10])

# ground truth
plt.subplot(121)
plt.imshow(I); plt.axis('off'); plt.title('ground truth')
annIds = cocoGt.getAnnIds(imgIds=imgId, catIds=[17])
anns = cocoGt.loadAnns(annIds)
cocoGt.showAnns(anns)