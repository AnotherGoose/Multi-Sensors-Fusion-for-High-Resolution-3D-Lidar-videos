# Pixel Distribution
This is an in-depth guide for the python scripts which dictate sampling allocation in the 
**Task-Based Adaptive Sampling** model

## Random Sampling
Below are the functions found within the **Random_Sampling.py**


## Uniform Sampling
Below are the functions found within the **Uniform_Sampling.py**

### uniformS(img, nPixels)
**uniformS** takes in an image and the number of new pixels to allocate and sub-samples the image uniformly. 
The distribution of pixels is based upon the aspect ratio of the original image. 
The function **uniformSpread** within **Distribution_Utils.py** handles this process.
#### Inputs
* **img** - openCV input image (depth map)
* **nPixels** - New amount of pixels to uniformly sub-sample the input image 
#### Outputs
* **US** - The newly uniformly sampled output (not-interpolated)

#### Example
![image](https://i.imgur.com/iwIgLRD.png)

### uniformAS(img, ROI, nPixels, rPort)
**uniformAS** takes in an image, the number of new pixels to sub-sample the image uniformly to, the Regions of Interest 
(ROI) and the portion for the ROI's. This function uniformly samples both the ROI and background. The difference between this function and uniformS is that the variable rPort deternimes how much of the pixels are uniformly allocated towards the ROI and the remainder to the background

#### Inputs
* **img** - openCV input image (depth map)
* **ROI** - 2D numpy array with the x, y, width and height of the ROI stored respectively
* **nPixels** - New amount of pixels to uniformly sub-sample the input image 
* **rPort** - The portion of the total sub-sample pixels to allocate to the ROI's (0 > n < 1)
#### Outputs

#### Example
<u>Bounding Box</u>

![image](https://i.imgur.com/ixdFQMe.png)

<u>Instance Segmentation (CURRENTLY NOT INSTANCE SEG</u>

![image](https://i.imgur.com/ixdFQMe.png)


## Metropolis Hastings
Below are the functions found within **Met_Hastings.py**

## Distribution Utilities
Below are the functions found within **Distribution_Utils.py**

explain individual functions and theory behind