import cv2
import numpy as np
import os
import sys

initialDir = os.path.abspath(os.getcwd())
os.chdir('../../')
projectDir = os.path.abspath(os.getcwd())

#Import Images
imgName = "Dad.png"

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
from Distribution_Utils import combineMasks
import Met_Hastings as M_H
#import Random_Sampling
#import Unifrom_Sampling

#Remove all detections lower than this confidence
cThresh = 0.5

#Removing all detections having overlap higher than this value
oThresh = 0.3

cuda = False

boundingROI = getYOLOPredsImg(img, cThresh, oThresh, cuda)
instanceROI, instanceMasks = getMRCNNPredsImg(img)

# Compress Masks
instanceMask = combineMasks(instanceMasks)

# Make this a user input
pixels = 10000

RWMHBB = M_H.RandomWalkMetHastingsBBox(depth, boundingROI, pixels, 1, 10, 100, 25)
RWMHI = M_H.RandomWalkMetHastingsInstance(depth, instanceMask, pixels, 1, 10, 1000, 25)

cv2.imshow('RWMH-Instance', RWMHI)
cv2.imshow('RWMH-BB', RWMHBB)

cv2.waitKey(0)
cv2.destroyAllWindows()