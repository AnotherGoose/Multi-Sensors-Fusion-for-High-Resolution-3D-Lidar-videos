import cv2
import numpy as np
import os
import sys
import pandas as pd
import scipy.io

initialDir = os.path.abspath(os.getcwd())
os.chdir('../../')
projectDir = os.path.abspath(os.getcwd())

#Import Images
#imgName = "Mannequin.png"
imgName = "frame_0003.png"

os.chdir("Input_RGB")
img = cv2.imread(imgName)

os.chdir("..")

os.chdir("Input_Depth")
depth = cv2.imread(imgName)
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

os.chdir(initialDir)

#Import Libraries
sys.path.append(os.path.join(initialDir, "YOLO"))
from YOLO import getYOLOPredsImg
sys.path.append(os.path.join(initialDir, "Mask_RCNN"))
from Mask_RCNN import getMRCNNPredsImg
sys.path.append(os.path.join(initialDir, "Pixel_Distribution"))

#Import point distribution libraries
import Distribution_Utils as utils
import Met_Hastings as M_H
#import Random_Sampling
#import Unifrom_Sampling

#Remove all detections lower than this confidence
cThresh = 0.5

#Removing all detections having overlap higher than this value
oThresh = 0.3

cuda = False

detectedBB, boundingROI, outputRecog = getYOLOPredsImg(img, cThresh, oThresh, cuda)
detectedI, instanceROI, instanceMasks, outputRecog = getMRCNNPredsImg(img)

# Compress Masks
instanceMask = utils.combineMasks(instanceMasks)

# Make this a user input
pixels = 5000
pointCloud = True
sheetName = "frame1"
initial = True


def savePoints(fileName, array):
    frames, dim, length = array.shape

    with open(fileName, 'wb') as f:  # need 'wb' in Python3
        for i in range(frames):
            name = 'frame' + str(i)
            x = array[i][0]
            y = array[i][1]
            z = array[i][2]
            scipy.io.savemat(f, {name: (x, y, z)})
    return 0

if __name__ == "__main__":
    if detectedBB:
        RWMHBB = M_H.RandomWalkMetHastingsBBox(depth, boundingROI, pixels, 1, 10, 100, 25)
        cv2.imshow('RWMH-BB', RWMHBB)
        pUsed = utils.nonNan(RWMHBB)
        interpolatedRWMHBB = utils.nInterp2D(pUsed, RWMHBB)
        cv2.imshow('InterpolatedBB', interpolatedRWMHBB)

        if pointCloud:
            frames = 0
            x, y, z = utils.seperateArrayPC(RWMHBB, pUsed)
            fileName = 'outputBB.mat'
            frames = np.zeros((1, 3, pUsed))
            frames[0] = x, y, z
            initial = False
            savePoints(fileName, frames)
    else:
        print("YOLO did not detect an object")

    if detectedI:
        RWMHI = M_H.RandomWalkMetHastingsInstance(depth, instanceMask, pixels, 1, 10, 1000, 25)
        cv2.imshow('RWMH-Instance', RWMHI)
        pUsed = utils.nonNan(RWMHI)
        interpolatedRWMHI = utils.nInterp2D(pUsed, RWMHI)
        cv2.imshow('InterpolatedInst', interpolatedRWMHI)

        if pointCloud:
            frames = 0
            x, y, z = utils.seperateArrayPC(RWMHI, pUsed)
            fileName = 'outputInst.mat'
            frames = np.zeros((1, 3, pUsed))
            frames[0] = x, y, z
            initial = False
            savePoints(fileName, frames)
    else:
        print("Mask-RCNN did not detect an object")

    cv2.waitKey(0)
    cv2.destroyAllWindows()