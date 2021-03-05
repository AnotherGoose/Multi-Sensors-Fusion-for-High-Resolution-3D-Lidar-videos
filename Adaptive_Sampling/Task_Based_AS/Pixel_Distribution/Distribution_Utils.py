import numpy as np
import math
import random
import cv2
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error

def walkIndex(prevI, max, sigma):
    a = -1
    b = 1
    if prevI - sigma < 0:
        #Can exceed the minimum
        a = -(prevI) / sigma
    if prevI + sigma > max:
        #Index can exceed maximum
        b = (max-prevI)/sigma
    propI = int(prevI + round(sigma * random.uniform(a, b)))
    return propI

def createFeatureMapBBox(img, ROI, bConst, rConst):
    H, W = img.shape
    fMap = np.zeros((H, W))

    #Background
    fMap[:, :] = bConst

    #ROI
    for r in ROI:
        x, y, w, h = r
        for i in range(y, y + h):
            for j in range(x, x + w):
                fMap[i][j] = rConst
    return fMap

def createFeatureMapInstance(mask, bConst, iConst):
    row, col = mask.shape
    fMap = np.zeros((row, col))

    fMap[:, :] = bConst

    for j in range(row):
        for k in range(col):
            if mask[j][k] == True:
                fMap[j][k] = iConst

    return fMap

def rmse(predictions, targets):
    RMSE = mean_squared_error(targets, predictions, squared=False)
    return RMSE

def seperateArrayInt(array, pixels):
    #Seperate the array for interpolation
    h, w = array.shape

    # Values for non NaN points in the array
    values = np.empty(pixels)
    # X and Y coordinates of values
    points = np.empty((pixels, 2))

    c = 0
    for i in range(h):
        for j in range(w):
            if not math.isnan(array[i][j]):
                values[c] = array[i][j]
                points[c] = (i, j)
                c += 1
    return values, points

def seperateArrayPC(array, pixels):
    #Seperate the array for Point Cloud
    h, w = array.shape

    # Values for non NaN points in the array
    x = np.empty(pixels)
    y = np.empty(pixels)
    z = np.empty(pixels)

    c = 0
    for i in range(h):
        for j in range(w):
            if not math.isnan(array[i][j]):
                x[c] = j
                y[c] = i
                z[c] = array[i][j]
                c += 1
    #x = x.astype(np.uint8)
    #y = y.astype(np.uint8)
    #z = z.astype(np.uint8)
    return x, y, z

def nInterp2D(pixels, array):
    # Given a specific number of non-NaN pixels
    # interpolate to the grid of the 2D array
    c = 0
    h, w = array.shape

    # Grid to interpolate over
    grid_y, grid_x = np.mgrid[0:h, 0:w]

    values, points = seperateArrayInt(array, pixels)

    Nearest = griddata(points, values, (grid_y, grid_x), method='nearest')
    Nearest = Nearest.astype(np.uint8)
    return Nearest

def getNewDimensions(nPixels, oWidth, oHeight):
    #Calculate Aspect Ratio
    aspectRatio = oWidth/oHeight

    #Calculate new width and height
    nWidth = math.sqrt(nPixels * aspectRatio)
    nHeight = nPixels/ nWidth

    return (nWidth, nHeight)

def checkRounding(limit, w, h):
    if round(w) * round(h) > limit:
        if w * round(h) > limit:
            h = math.floor(h)
        if round(w) * h > limit:
            w = math.floor(w)

    w = round(w)
    h = round(h)
    return w, h

def uniformSpread(oArray, nPoints, nArray):
    oH, oW = oArray.shape

    if nPoints > oH * oW:
        nPoints = oH * oW

    nW, nH = getNewDimensions(nPoints, oW, oH)
    nW, nH = checkRounding(nPoints, nW, nH)

    stepW = np.linspace(0, oW - 1, nW)
    stepH = np.linspace(0, oH - 1, nH)

    counter = 0
    for i in stepH:
        i = round(i)
        for j in stepW:
            j = round(j)
            nArray[i][j] = oArray[i][j]
            counter += 1
    return nArray

def nonNan(array):
    h, w = array.shape
    counter = 0
    for i in range(h):
        for j in range(w):
            if not math.isnan(array[i][j]):
                counter += 1
    return counter

def invertGrayscale(img):
    img = cv2.bitwise_not(img)
    return img

def combineMasks(masks):
    row, col, width = masks.shape
    mask = np.zeros((row, col), dtype=bool)
    for i in range(width):
        mask = np.logical_or(mask, masks[:, :, i])
    return mask