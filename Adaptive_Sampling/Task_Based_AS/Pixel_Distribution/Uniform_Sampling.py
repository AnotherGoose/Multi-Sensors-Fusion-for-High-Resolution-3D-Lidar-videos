import numpy as np
import Distribution_Utils as utils

def uniformS(img, nPixels):
    imH, imW = img.shape

    if(nPixels > imH * imW):
        nPixels = imH * imW

    US = np.empty((imH, imW))
    US[:] = np.nan

    US = utils.uniformSpread(img, nPixels, US)

    uniformNearest = utils.nInterp2D(nPixels, US)
    return uniformNearest



def uniformAS(img, ROI, nPixels, rPort):
    imH, imW = img.shape

    if(nPixels > imH * imW):
        nPixels = imH * imW

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    #Can be an input
    bPort = 1 - rPort

    rPixels = round(nPixels * rPort)
    bPixels = round(nPixels * bPort)

    #Pixels Remaining
    pRem = 0

    #Total pixels in ROI
    roiSum = 0
    for r in ROI:
        x, y, w, h = r
        roiSum += ((w + 1) * (h + 1))

    for r in ROI:
        x, y, w, h = r
        roiPort = ((w + 1) * (h + 1))/roiSum
        roiPixels = round(roiPort * rPixels)

        #Add on the remainding pixels
        roiPixels += pRem
        AS[y:y+h, x:x+w] = utils.uniformSpread(img[y:y+h, x:x+w], roiPixels, AS[y:y+h, x:x+w])

        pRem = roiPixels - utils.nonNan(AS[y:y+h, x:x+w])

    bPixels += pRem
    AS = utils.uniformSpread(img, bPixels, AS)
    pUsed = utils.nonNan(AS)
    uniformNearest = utils.nInterp2D(pUsed, AS)
    #uniformNearest = AS
    return uniformNearest

'''
import cv2
ROI = np.array([[0, 0, 142, 142]])
depth = cv2.imread("Mannequin.png")
depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
AS = uniformAS(depth, ROI, 10000, 0.6)
cv2.imshow("UAS", AS)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''