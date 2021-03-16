import numpy as np
import random
import Distribution_Utils as utils
import math

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
    return AS

def RandomWalkMetHastingsBBox(img, ROI, pixels, bConst, roiConst, sigma, N):
    imH, imW = img.shape

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    AS = utils.uniformSpread(img, pixels, AS)
    fMap = utils.createFeatureMapBBox(img, ROI, bConst, roiConst)

    RWMH = RandomWalkMetHastings(img, AS, fMap, sigma, N)
    return RWMH

def RandomWalkMetHastingsInstance(img, mask, pixels, bConst, roiConst, sigma, N):
    imH, imW = img.shape

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    AS = utils.uniformSpread(img, pixels, AS)
    fMap = utils.createFeatureMapInstance(mask, bConst, roiConst)

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


if __name__ == "__main__":
    import cv2
    ROI = np.array([[27, 8, 95, 127]])
    depth = cv2.imread("Mannequin.png")
    depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)

    pixels = 10000

    RMWH = RandomWalkMetHastingsBBox(depth, ROI, pixels, 1, 10, 100, 25)
    MH = MetHastingsBBox(depth, ROI, pixels, 1, 10, 5)

    row, col = depth.shape

    for j in range(row):
        for k in range(col):
            if not math.isnan(RMWH[j][k]):
                RMWH[j][k] = 1
            if not math.isnan(MH[j][k]):
                MH[j][k] = 1

    cv2.imshow("RMWH", RMWH)
    cv2.imshow("MH", MH)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



