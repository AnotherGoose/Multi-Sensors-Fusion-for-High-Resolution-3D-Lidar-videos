import cv2
import numpy as np
import os
import sys
import scipy.io
from imutils.video import FPS

#Change directory for Matlab Script
baseName = os.path.basename(__file__)
dir = __file__
dir = dir[:-(len(baseName))]

os.chdir(dir)
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

# Make Definitions
boundingBox = 0
instanceSegmentation = 1
randomSampling = 2
uniformSampling = 3

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

def is_cuda_cv(): # 1 == using cuda, 0 = not using cuda
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return 1
        else:
            return 0
    except:
        return 0

def videoDetection(inputRGBVideoPath, inputDepthVideoPath, outputDepthPath, outputRecogPath, pixels, detectionType, pointCloud, displayOutput):
    # detectionType   0 - Bounding box
    #                 1 - Instance segmentation

    #Thresholds for detection
    cThresh = 0.5
    oThresh = 0.3

    capRGB = cv2.VideoCapture(inputRGBVideoPath)
    capDepth = cv2.VideoCapture(inputDepthVideoPath)


    #Check if the input RGB video and depth video is opened properly
    if (capRGB.isOpened() == False) or (capDepth.isOpened() == False):
        print("Error opening video stream or file")
        return

    (successRGB, img) = capRGB.read()
    (successDepth, depth) = capDepth.read()

    depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    vidDepth = cv2.VideoWriter(outputDepthPath, fourcc, capDepth.get(cv2.CAP_PROP_FPS), (depth.shape[1], depth.shape[0]), 0)

    vidRecog = cv2.VideoWriter(outputRecogPath, fourcc, capDepth.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]), True)

    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

    cuda = is_cuda_cv()
    if cuda == 1:
        print("OpenCV CUDA Enabled")

    initial = True
    frames = 0
    fCount = 1

    yoloConfigPath = dir + 'YOLO/Model_Data/yolov4.cfg'
    yoloWeightsPath = dir + 'YOLO/Model_Data/yolov4.weights'

    if detectionType == boundingBox:
        net = cv2.dnn.readNetFromDarknet(yoloConfigPath, yoloWeightsPath)
        if cuda == 1:
            print("OpenCV CUDA Enabled")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    while successRGB and successDepth:
        fps = FPS().start()
        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)


        # Fix for when pixels and pUsed don't match up for saving pClouds
        pUsed = utils.nonNan(U_S.uniformS(depth, pixels))
        # End Fix

        if detectionType == boundingBox:
            # Bounding Box
            detected, boundingROI, outputRecog = getYOLOPredsImg(img, cThresh, oThresh, net)
            fName = 'outputBB.mat'
            if detected:
                #outputDepth = M_H.RandomWalkMetHastingsBBox(depth, boundingROI, pixels, 1, 10, 100, 25)
                #outputDepth = M_H.RandomWalkMetHastingsBBox(depth, boundingROI, pixels, 1, 10, 1000, 25)
                #Uniformly Sample the ROI at 80% (Use the same pixels as in uniform sampling to enusure the pixels for random sampling and uniform adaptive sampling are the same)
                outputDepth = M_H.AdaptiveRandomWalkMetHastingsBBox(depth, boundingROI, pUsed, 0.6)
            else:
                outputDepth = R_S.randomS(depth, pUsed)
                outputRecog = img

        elif detectionType == instanceSegmentation:
            # Instance Segmentation
            detected, instanceROI, instanceMasks, outputRecog = getMRCNNPredsImg(img)
            fName = 'outputInst.mat'
            if detected:
                # Compress Masks
                instanceMask = utils.combineMasks(instanceMasks)
                outputDepth =  M_H.RandomWalkMetHastingsInstance(depth, instanceMask, pixels, 1, 10, 1000, 25)
            else:
                #If there is no object detected, preform random sampling
                outputDepth = R_S.randomS(depth, pUsed)
                outputRecog = img

        fps.stop()

        interOutputDepth = utils.nInterp2D(pUsed, outputDepth)

        if pointCloud:
            x, y, z = utils.seperateArrayPC(outputDepth, pUsed)
        else:
            pUsed = utils.nonNan(interOutputDepth)
            x, y, z = utils.seperateArrayPC(interOutputDepth, pUsed)

        fileName = projectDir + '/Sampling_Output/' + fName
        if initial:
            frames = np.zeros((1, 3, pUsed))
            frames[0] = x, y, z
            initial = False
        else:
            newDim = np.vstack([x, y, z])
            newDim = newDim[np.newaxis, :, :]
            frames = np.append(frames, newDim, axis=0)

        #h, w = outputDepth.shape



        if displayOutput:
            cv2.imshow("ModelOutput", interOutputDepth)
            cv2.imshow("Predictions", outputRecog)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break the loop
            if key == ord("q"):
                break

        vidDepth.write(interOutputDepth)
        vidRecog.write(outputRecog)

        (successDepth, depth) = capDepth.read()
        (successRGB, img) = capRGB.read()

        print("Frame {:d} sample time: {:.2f}s".format(fCount, fps.elapsed()))
        fCount += 1
    #if pointCloud:
        #Save points if there is a point cloud
        savePoints(fileName, frames)

    capDepth.release()
    capRGB.release()
    vidDepth.release()
    vidRecog.release()
    if displayOutput:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":

    inputDepth = projectDir + '/Input_Depth/DLR.mp4.mp4'
    inputRGB = projectDir + '/Input_RGB/RLR.mp4.mp4'
    outputDepthPath = projectDir + '/Sampling_Output/depthOutputHN.mp4'
    outputRecogPath = projectDir + '/Sampling_Output/recogOutput.mp4'
    detectionMethod = boundingBox
    print(inputDepth)
    print(inputRGB)
    print(outputDepthPath)
    print(outputRecogPath)
    pCloud = True
    prec = 10000
    displayOutput = True

    os.chdir(initialDir)
    videoDetection(inputRGB, inputDepth, outputDepthPath, outputRecogPath, prec, detectionMethod, pCloud, displayOutput)
