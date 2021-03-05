import cv2
import numpy as np
import os
import sys
import math
import scipy.io

initialDir = os.path.abspath(os.getcwd())
os.chdir('../../')
projectDir = os.path.abspath(os.getcwd())

#Import Libraries
sys.path.append(os.path.join(initialDir, "YOLO"))
from YOLO import getYOLOPredsImg
sys.path.append(os.path.join(initialDir, "Mask_RCNN"))
from Mask_RCNN import getMRCNNPredsImg
sys.path.append(os.path.join(initialDir, "Pixel_Distribution"))
import Distribution_Utils as utils
import Met_Hastings as M_H
import Random_Sampling as R_S
import Uniform_Sampling as U_S

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

def videoDetection(inputRGBVideoPath, inputDepthVideoPath, outputDepthPath, outputRecogPath,detectionType, pointCloud,pixels):
    # detectionType   0 - Bounding box
    #                 1 - Instance segmentation

    cThresh = 0.5
    oThresh = 0.3

    cuda = False

    capRGB = cv2.VideoCapture(inputRGBVideoPath)
    capDepth = cv2.VideoCapture(inputDepthVideoPath)

    iter = 1

    if (capRGB.isOpened() == False) or (capDepth.isOpened() == False):
        print("Error opening video stream or file")
        return

    (successRGB, img) = capRGB.read()
    (successDepth, depth) = capDepth.read()

    depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

    vidDepth = cv2.VideoWriter(outputDepthPath, cv2.VideoWriter_fourcc(
        *"MJPG"), capDepth.get(cv2.CAP_PROP_FPS), (depth.shape[1], depth.shape[0]), 0)

    vidRecog = cv2.VideoWriter(outputRecogPath, cv2.VideoWriter_fourcc(
        *"MJPG"), capDepth.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]), True)

    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

    initial = True
    frames = 0

    while successRGB and successDepth:
        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
        if detectionType == 0:
            # Bounding Box
            detected, boundingROI, outputRecog = getYOLOPredsImg(img, cThresh, oThresh, cuda)
            if detected:
                outputDepth = M_H.RandomWalkMetHastingsBBox(depth, boundingROI, pixels, 1, 10, 100, 25)
            else:
                #Fix for when pixels and pUsed dont match up for saving pClouds
                pUsed = utils.nonNan(U_S.uniformS(depth, pixels))
                #End Fix

                outputDepth = R_S.randomS(depth, pUsed)
                outputRecog = img

            if pointCloud:
                pUsed = utils.nonNan(outputDepth)
                x, y, z = utils.seperateArrayPC(outputDepth, pUsed)
                fileName = 'outputBB.mat'
                if initial:
                    frames = np.zeros((1, 3, pUsed))
                    frames[0] = x, y, z
                    initial = False
                else:
                    newDim = np.vstack([x, y, z])
                    newDim = newDim[np.newaxis, :, :]
                    frames = np.append(frames, newDim, axis=0)

        elif detectionType == 1:
            # Instance
            detected, instanceROI, instanceMasks, outputRecog = getMRCNNPredsImg(img)

            if detected:
                # Compress Masks
                instanceMask = utils.combineMasks(instanceMasks)
                outputDepth =  M_H.RandomWalkMetHastingsInstance(depth, instanceMask, pixels, 1, 10, 1000, 25)
            else:
                #Fix for when pixels and pUsed dont match up for saving pClouds
                pUsed = utils.nonNan(U_S.uniformS(depth, pixels))
                #End Fix

                outputDepth = R_S.randomS(depth, pUsed)
                outputRecog = img

            if pointCloud:
                pUsed = utils.nonNan(outputDepth)
                x, y, z = utils.seperateArrayPC(outputDepth, pUsed)
                fileName = 'outputInst.mat'
                if initial:
                    frames = np.zeros((1, 3, pUsed))
                    frames[0] = x, y, z
                    initial = False
                else:
                    newDim = np.vstack([x, y, z])
                    newDim = newDim[np.newaxis, :, :]
                    frames = np.append(frames, newDim, axis=0)

        elif detectionType == 2:
            # Random Sampling
            outputDepth = R_S.randomS(depth, pixels)
            outputRecog = img

        elif detectionType == 3:
            # Uniform Sampling
            outputDepth = U_S.uniformS(depth, pixels)
            outputRecog = img

        h, w = outputDepth.shape

        # Values for non NaN points in the array
        values = np.empty(pixels)
        # X and Y coordinates of values
        points = np.empty((pixels, 2))

        c = 0
        for i in range(h):
            for j in range(w):
                if math.isnan(outputDepth[i][j]):
                    outputDepth[i][j] = 0

        outputDepth = outputDepth.astype(np.uint8)

        cv2.imshow("ModelOutput", outputDepth)
        cv2.imshow("Predictions", outputRecog)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break the loop
        if key == ord("q"):
            break

        iter += 1

        vidDepth.write(outputDepth)
        vidRecog.write(outputRecog)

        (successDepth, depth) = capDepth.read()
        (successRGB, img) = capRGB.read()

    savePoints(fileName, frames)

    capDepth.release()
    capRGB.release()
    vidDepth.release()
    vidRecog.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    inputDepth = 'Input_Depth/Frames_15fps.mp4'
    inputRGB = 'Input_RGB/Frames_15fps.mp4'
    outputDepthPath = 'Output/depthOutput.avi'
    outputRecogPath = 'Output/recogOutput.avi'
    os.chdir(initialDir)
    videoDetection(inputRGB, inputDepth, outputDepthPath, outputRecogPath, 0, True, 5000)
