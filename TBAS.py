import os
import sys

#Change directory for Matlab Script
baseName = os.path.basename(__file__)
dir = __file__
dir = dir[:-(len(baseName))]

print(dir)
#os.chdir(dir)
pDir = os.path.abspath(os.getcwd())
os.chdir('Adaptive_Sampling/Task_Based_AS/')
iDir = os.path.abspath(os.getcwd())

#Import Libraries
sys.path.append(os.path.join(iDir))
from depthVideo import videoDetection

# Make Definitions
boundingBox = 0
instanceSegmentation = 1
randomSampling = 2
uniformSampling = 3

inputDepth = pDir + '/Input_Depth/NoisyFramesHN_15fps.mp4'
inputRGB = pDir + '/Input_RGB/Frames_15fps.mp4'
outputDepthPath = pDir + '/Sampling_Output/depthOutputHN.mp4'
outputRecogPath = pDir + '/Sampling_Output/recogOutput.mp4'
pCloud = True

#Recognition Input
recognition = "Placeholder"
while recognition != "0" or recognition != "1":
    recognition = input("Enter which image recognition method to employ: \n0 - Bounding Box \n1 - Instance Segmentation\n")
    if recognition == "0":
        print("Employing Bounding Box recognition")
        break
    if recognition == "1":
        print("Employing Instance Segmentation recognition")
        break
    if recognition != "0" or recognition != "1":
        print("Please input a valid number")
recognition = int(recognition)

#Collect pixel input
uInput = "Placeholder"
valid = False
while valid != True:
    uInput = int(input("How many points of the video would you like to sample "))
    if uInput < 1:
        print("Please input a value greater than 0")
    else:
        valid = True
pixels = int(uInput)

#Display Input
uInput = "Placeholder"
while uInput != "0" or uInput != "1":
    uInput = input("Would you like to display the process: \n0 - No \n1 - Yes\n")
    if uInput == "0":
        print("Hiding Process")
        displayOutput = False
        break
    if uInput == "1":
        print("Showing Process")
        displayOutput = True
        break
    if uInput != "0" or uInput != "1":
        print("Please input a valid number")


videoDetection(inputRGB, inputDepth, outputDepthPath, outputRecogPath, pixels, recognition, pCloud, displayOutput)