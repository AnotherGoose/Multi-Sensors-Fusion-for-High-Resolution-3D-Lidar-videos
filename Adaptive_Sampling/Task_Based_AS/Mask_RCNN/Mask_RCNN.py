import cv2
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize_cv
from coco import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def getMRCNNPredsImg(img):

    os.chdir("Mask_RCNN")


    currDir = os.path.abspath(os.getcwd())
    modelDir = os.path.join(currDir, "logs")
    cocoDir = os.path.join(currDir, "Model_Data/mask_rcnn_coco.h5")

    if not os.path.exists(cocoDir):
        utils.download_trained_weights(cocoDir)

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=modelDir, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(cocoDir, by_name=True)

    # Read off text file
    classNames = ["Dumb", "Bitch"]

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    # Run detection
    results = model.detect([img], verbose=1)

    # Visualize results
    r = results[0]

    display = True

    if display == True:
        InstSeg = visualize_cv.displayInstances(img, r['rois'], r['masks'], r['class_ids'],
                                                class_names, r['scores'])

    cv2.imshow('Instance Segmentation', InstSeg)

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
    return ROI, masks