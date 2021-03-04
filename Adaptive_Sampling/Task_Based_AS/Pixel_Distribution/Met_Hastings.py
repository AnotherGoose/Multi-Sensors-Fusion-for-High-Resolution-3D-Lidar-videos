import numpy as np
import math
import random
import Distribution_Utils as utils


def MetHastingsBBox(img, ROI, pixels, bConst, roiConst, N):
    fMap = utils.createFeatureMapBBox(img, ROI, bConst, roiConst)
    MH = MetHastings(img, pixels, fMap, N)
    return MH

def MetHastingsInstance(img, mask, pixels, bConst, roiConst, N):
    fMap = utils.createFeatureMapInstance(mask, bConst, roiConst)
    MH = MetHastings(img, pixels, fMap, N)
    return MH

def MetHastings(img, pixels, fMap, N):
    imH, imW = img.shape

    #Define AS value array and Feature Map
    #fMap = np.zeros((imH, imW))
    AS = np.empty((imH, imW))
    AS[:] = np.nan

    #Pixels sampled
    pCount = 0

    #Set initial Met Hastings position
    rX = random.randint(0, imW - 1)
    rY = random.randint(0, imH - 1)
    nX = rX
    nY = rY
    n = img[rY][rX]
    AS[rY][rX] = n
    pCount += 1

    #Determine if the iteration succeeded in finding a good candidate
    accept = False

    #Loop through other pixels
    while pCount < pixels:
        accept = False
        for i in range(N):
            # Random x and y Values
            rX = random.randint(0, imW - 1)
            rY = random.randint(0, imH - 1)

            # Ratio of new point compared to previous on feature map
            α = min((fMap[rY][rX]) / (fMap[nY][nX]), 1)

            # Random int between 1 and 0
            r = random.uniform(0, 1)
            if r < α:
                # Check if pixel is used
                if math.isnan(AS[rY][rX]):
                    nX = rX
                    nY = rY
                    n = img[rY][rX]
                    accept = True

        if accept:
            # Check if pixel is used
            if math.isnan(AS[nY][nX]):
                AS[nY][nX] = n
                pCount += 1

    nearestAS = utils.nInterp2D(pixels, AS)
    #nearestAS = AS
    return nearestAS

def RandomWalkMetHastingsBBox(img, ROI, pixels, bConst, roiConst, sigma, N):
    imH, imW = img.shape

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    AS = utils.uniformSpread(img, pixels, AS)
    fMap = utils.createFeatureMapBBox(img, ROI, bConst, roiConst)

    RWMH = RandomWalkMetHastings(img, AS, fMap, sigma, N)
    return RWMH

def RandomWalkMetHastingsInstance(img, mask, pixels, bConst, iConst, sigma, N):
    imH, imW = img.shape

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    AS = utils.uniformSpread(img, pixels, AS)
    fMap = utils.createFeatureMapInstance(mask, bConst, iConst)

    RWMH = RandomWalkMetHastings(img, AS, fMap, sigma, N)
    return RWMH

def RandomWalkMetHastings(img, AS, fMap, sigma, N):
    imH, imW = img.shape

    #Split array to quickly go through pixels
    pUsed = utils.nonNan(AS)
    values, points = utils.seperateArrayInt(AS, pUsed)

    for i in range(values.size):
        y, x = points[i]
        yPrev = y = int(y)
        xPrev = x = int(x)
        for j in range(N):
            xProp = utils.walkIndex(xPrev, imW-1, sigma)
            yProp = utils.walkIndex(yPrev, imH-1, sigma)

            # Ratio of new point compared to previous on feature map
            α = min((fMap[yProp][xProp]) / (fMap[yPrev][xPrev]), 1)

            # Random int between 1 and 0
            r = random.uniform(0, 1)
            # Check proposal
            if r < α:
                # Check if point is used
                if math.isnan(AS[yProp][xProp]):
                    yPrev = yProp
                    xPrev = xProp
        AS[y][x] = np.nan
        AS[yPrev][xPrev] = img[yPrev][xPrev]
    nearestAS = AS
    return nearestAS

'''
import cv2
import os
import sys
ROI = np.array([[0, 0, 142, 142]])
x,y,w,h = ROI[0]
depth = cv2.imread("Mannequin.png")
depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
pixels = 10000
RWMH = RandomWalkMetHastingsBBox(depth, ROI, pixels, 1, 10, 100, 25)
MH = MetHastingsBBox(depth, ROI, pixels, 1, 10, 5)
#print(utils.rmse(AS, depth))
#print(utils.rmse(AS[y:y + h, x:x + w], depth[y:y + h, x:x + w]))
cv2.imshow("RWMH", RWMH)
cv2.imshow("MH", MH)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''



