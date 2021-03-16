# Pixel Distribution
This is an in-depth guide for the python scripts which dictate sampling allocation in the 
**Task-Based Adaptive Sampling** model

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
* **US** - The new uniformly sub-sampled output (not-interpolated)

#### Example

Bounding Box            |
:-------------------------:|
![](https://i.imgur.com/iwIgLRD.png)  | 


### uniformAS(img, ROI, nPixels, rPort)
**uniformAS** takes in an image, the number of new pixels to sub-sample the image uniformly to, the Regions of Interest 
(ROI) and the portion for the ROI's. This function uniformly samples both the ROI and background. The difference between this function and uniformS is that the variable rPort determines how much of the pixels are uniformly allocated towards the ROI and the remainder to the background

#### Inputs
* **img** - openCV input image (depth map)
* **ROI** - 2D numpy array with the x, y, width and height of the ROI stored respectively
* **nPixels** - New amount of pixels to uniformly sub-sample the input image 
* **rPort** - The portion of the total sub-sample pixels to allocate to the ROI's (0 > n < 1)
#### Outputs
* **AS** - The new uniformly adaptively sub-sampled output (not-interpolated)

#### Example
Bounding Box            | Instance Segmentation
:-------------------------:|:-------------------------:
![](https://i.imgur.com/ixdFQMe.png)  | There is no instance segmentation for uniform adaptive sampling because, <br>in a real-life scenario, sampling uniformly can miss periodicity in shapes. <br>Due to this issue, I ignored implementation for Instance Segmentation


## Random Sampling
Below are the functions found within the **Random_Sampling.py**

### randomS(img, pixels)
**randomS** takes in an image and the number of new pixels to allocate and sub-samples the image randomly. 
The indexes for the new points are randomly chosen. Further points are allocated until all pixels are used up. There is a check to determine if the point has been used before or not, checking if the pixel is a NaN. 

#### Inputs
* **img** - openCV input image (depth map)
* **pixels** - New amount of pixels to randomly sub-sample the input image 
#### Outputs
* **RS** - The new randomly sub-sampled output (not-interpolated)
#### Example
Bounding Box            | 
:-------------------------:|
![](https://i.imgur.com/4FsshD8.png)  |

### randomAS(img, ROI, pixels, roiPort)
**randomAS** takes in an image, the number of new pixels to sub-sample the image randomly to, the Regions of Interest 
(ROI) and the portion for the ROI's. This function randomly samples both the ROI and background. The difference between this function and randomS is that the variable roiPort determines how much of the pixels are randomly allocated towards the ROI and the remainder to the background.


#### Example
Bounding Box            | Instance Segmentation
:-------------------------:|:-------------------------:
![](https://i.imgur.com/Nh0Bk3Y.png)  | As the preformance of all random adaptive sample algorithms proposed <br>preform similarly only Metropolis Hastings proposals were adapted for <br>instance segmentation 

## Metropolis Hastings
Below are the functions found within **Met_Hastings.py**

### MetHastingsBBox(img, ROI, pixels, bConst, roiConst, N)
**MetHastingsBBox** is the setup function to generate a feature map for the Metropolis Hastings algorithm for a **Bounding Box** adaptation. The feature map function is found within distribution utils.
#### Inputs
* **img** - openCV input image (depth map)
* **ROI** - 2D numpy array with the x, y, width and height of the ROI stored respectively
* **pixels** - New amount of pixels to sub-sample the input image by
* **bConst** - background constant weight for feature map 
* **roiConst** - ROI constant weight for feature map (detected objects) 
* **N** - Iterations to provide to Met Hastings algorithm for number of proposal points for each new index found
#### Outputs
* **MH** - - The new sub-sampled output (not-interpolated)

### MetHastingsInstance(img, mask, pixels, bConst, roiConst, N)
**MetHastingsBBox** is the setup function to generate a feature map for the Metropolis Hastings algorithm for an **Instance Segmentation** adaptation. The feature map function is found within distribution utils.
#### Inputs
* **img** - openCV input image (depth map)
* **mask** - A mask defined by Mask-RCNN defining areas either true of false depending if an object is detected within that pixel or not
* **pixels** - New amount of pixels to sub-sample the input image by
* **bConst** - background constant weight for feature map 
* **roiConst** - ROI constant weight for feature map (detected objects) 
* **N** - Iterations to provide to Met Hastings algorithm for number of proposal points for each new index found
#### Outputs
* **MH** - The new sub-sampled output (not-interpolated)

### MetHastings(img, pixels, fMap, N)
#### Inputs
* **img**
* **AS**
* **fMap**
* **N**
#### Outputs
* **AS**
#### Example

Bounding Box            |  Instance Segmentation
:-------------------------:|:-------------------------:
![](https://i.imgur.com/XxyqYDi.png)  |  ![](https://i.imgur.com/RaARUA0.png)

### RandomWalkMetHastingsBBox(img, ROI, pixels, bConst, roiConst, sigma, N)
**RandomWalkMetHastingsBBox** is the setup function to generate a feature map and a uniform spread of pixels for the Random Walk Metropolis Hastings algorithm for an **Bounding Box** adaptation. The feature map and uniform spread function is found within distribution utils.
#### Inputs
* **img** - openCV input image (depth map)
* **ROI** - 2D numpy array with the x, y, width and height of the ROI stored respectively
* **pixels** - New amount of pixels to sub-sample the input image by
* **bConst** - background constant weight for feature map 
* **roiConst** - ROI constant weight for feature map (detected objects)
* **sigma** - The variance of the new proposals from the orginal position (The amount of 'drift' points have)
* **N** - Iterations to provide to Radnom Walk Met Hastings algorithm for number of proposal points for each individual index
#### Outputs
* **RWMH** - The new sub-sampled output (not-interpolated)

### RandomWalkMetHastingsInstance(img, mask, pixels, bConst, roiConst, sigma, N)
**RandomWalkMetHastingsInstance** is the setup function to generate a feature map and a uniform spread of pixels for the Random Walk Metropolis Hastings algorithm for an **Instance segmentation** adaptation. The feature map and uniform spread function is found within distribution utils.
#### Inputs
* **img** - openCV input image (depth map)
* **mask** - A mask defined by Mask-RCNN defining areas either true of false depending if an object is detected within that pixel or not
* **pixels** - New amount of pixels to sub-sample the input image by
* **bConst** - background constant weight for feature map 
* **roiConst** - ROI constant weight for feature map (detected objects)
* **sigma** - The variance of the new proposals from the original position (The amount of 'drift' points have)
* **N** - Iterations to provide to Random Walk Met Hastings algorithm for number of proposal points for each individual index
#### Outputs
* **RWMH** - The new sub-sampled output (not-interpolated)

### RandomWalkMetHastings(img, AS, fMap, sigma, N)
**RandomWalkMetHastings**  implements the adaptation of the statistical distribution Metropolis Hastings dubbed Random Walk Metropolis Hastings.
This process is where the scene is uniformly sampled, and proposal points are compared against the previous point proposal on the feature map, which is weighted appropriately and either accepted or rejected, depending on a random variable.
This results in the uniformly sampled points being 'drifted' towards regions of interest. The 'drifting' of these points are dependant on a variance provided.

The following psuedo code outlines the execution of this method:
* Uniformly sample the scene
* Create a feature map of weighting ROI and background appropriately.
* Start a loop through all uniformly sampled points.
    * Loop N number of iterations
        * New proposal for this uniformly sampled point is:<br>
          n’ = n + round(σ*randn(1,1))
        * α = featuremap(n’)/featuremap(n)
	    * r is a uniform number between 0 and 1
	    * If r < α accept this new proposal <br>
        (n’ = n)<br>

Where: <br>
* **N** is the number of proposals to investigate for each uniformly sampled point<br>
* **n** is the previous point<br>
* **n’** is the new point proposal<br>
* **α** is the ration between the new and old point within the feature map<br>
* **σ** is the variance of the new proposals from the original position.


#### Inputs
* **img** - openCV input image (depth map)
* **AS** - Uniformly sub-sampled scene
* **fMap** - Feature map which has been pre-weighted
* **sigma** - The variance of the new proposals from the original position (The amount of 'drift' points have)
* **N** - Iterations to provide to Random Walk Met Hastings algorithm for number of proposal points for each individual index
#### Outputs
* **AS** - The new sub-sampled output (not-interpolated)
#### Example
Bounding Box            |  Instance Segmentation
:-------------------------:|:-------------------------:
![](https://i.imgur.com/FfzvicK.png)  |  ![](https://i.imgur.com/d6rGUF0.png)

## Distribution Utilities
Below are the functions found within **Distribution_Utils.py**

explain individual functions and theory behind