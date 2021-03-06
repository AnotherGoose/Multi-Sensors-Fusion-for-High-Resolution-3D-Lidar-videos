
import os
import csv

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize_cv
from coco import coco


baseName = os.path.basename(__file__)
dir = __file__
dir = dir[:-(len(baseName))]

os.chdir(dir)
cDir = os.path.abspath(os.getcwd())

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def getMRCNNPredsImg(img):


    #currDir = os.path.abspath(os.getcwd())
    modelDir = os.path.join(cDir, "logs")
    cocoDir = os.path.join(cDir, "Model_Data/mask_rcnn_coco.h5")

    if not os.path.exists(cocoDir):
        utils.download_trained_weights(cocoDir)

    config = InferenceConfig()




    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=modelDir, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(cocoDir, by_name=True)

    classesPath = cDir + "/Model_Data/classNames.txt"
    # Read off text file
    with open(classesPath, newline='') as csvfile:
        fileData = list(csv.reader(csvfile, delimiter=','))

    #Flatten File Data
    classNames = []
    for lists in fileData:
        for val in lists:
            classNames.append(val)

    # Run detection
    results = model.detect([img], verbose=1)


    # Visualize results
    r = results[0]

    display = True

    preds = 0
    if display == True:
        preds = visualize_cv.displayInstances(img, r['rois'], r['masks'], r['class_ids'],
                                                classNames, r['scores'])

    # rois are y1, x1 and y2, x2
    masks = r['masks'][:, :, :]
    ROI = r['rois'][:, :]

    depth, imH, imW = img.shape
    row, col = ROI.shape

    for i in range(row):
        # Convert ROI to YOLO implementation
        y1, x1, y2, x2 = ROI[i]
        if y1 < 0:
            y1 = 0
        if y2 > imH:
            y2 = imH - 1
        if x1 < 0:
            x1 = 0
        if x2 > imW:
            x2 = imW - 1
        w = x2 - x1
        h = y2 - y1
        ROI[i] = [x1, y1, w, h]



    os.chdir("../")

    #Check if anything was detected
    try:
        x, y, w, h = ROI[0]
        detected = True
    except:
        detected = False

    return detected, ROI, masks, preds