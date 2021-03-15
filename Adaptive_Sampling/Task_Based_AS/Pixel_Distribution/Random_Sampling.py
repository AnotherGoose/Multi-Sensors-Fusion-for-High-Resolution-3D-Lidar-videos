import numpy as np
import math
import random

def randomAS(img, ROI, pixels, roiPort):
    imH, imW = img.shape

    AS = np.empty((imH, imW))
    AS[:] = np.nan

    if pixels > imH * imW:
        print("Error: Pixel allocation is too large")
        pixels = imH * imW

    # Round ROI pixels
    newROIP = round(pixels * roiPort)

    totROI = 0

    #Total ROI pixels
    for r in ROI:
        x, y, w, h = r
        totROI += w * h

    if newROIP > totROI:
        newROIP = totROI

    pCount = 0
    for r in ROI:
        x, y, w, h = r
        #Portion of total ROI
        ROIPort = (w * h) / (totROI)
        nPixels = newROIP * ROIPort
        while pCount < nPixels:
            rX = random.randint(x, x + w - 1)
            rY = random.randint(y, y + h - 1)

            if math.isnan(AS[rY][rX]):
                AS[rY][rX] = img[rY][rX]
                pCount += 1

    while pCount < pixels:
        rX = random.randint(0, imW - 1)
        rY = random.randint(0, imH - 1)

        if math.isnan(AS[rY][rX]):
            AS[rY][rX] = img[rY][rX]
            pCount += 1
    return AS

def randomS(img, pixels):
    imH, imW = img.shape

    RS = np.empty((imH, imW))
    RS[:] = np.nan

    if pixels > imH * imW:
        print("Error: Pixel allocation is too large")
        pixels = imH * imW

    pCount = 0
    while pCount < pixels:
        rX = random.randint(0, imW - 1)
        rY = random.randint(0, imH - 1)

        if math.isnan(RS[rY][rX]):
            RS[rY][rX] = img[rY][rX]
            pCount += 1
    return RS

if __name__ == "__main__":
    import cv2
    import math
    ROI = np.array([[27, 8, 95, 127]])
    depth = cv2.imread("Mannequin.png")
    depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
    RAS = randomAS(depth, ROI, 10000, 0.6)
    RS = randomS(depth, 10000)

    row, col = depth.shape

    for j in range(row):
        for k in range(col):
            if not math.isnan(RAS[j][k]):
                RAS[j][k] = 1
            if not math.isnan(RS[j][k]):
                RS[j][k] = 1

    cv2.imshow("RAS", RAS)
    cv2.imshow("RS", RS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()