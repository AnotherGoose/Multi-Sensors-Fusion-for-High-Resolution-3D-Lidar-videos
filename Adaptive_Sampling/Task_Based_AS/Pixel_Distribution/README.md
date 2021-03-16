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
* **img** - OpenCV input image (depth map)
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
* **img** - OpenCV input image (depth map)
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
* **img** - OpenCV input image (depth map)
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
![](https://i.imgur.com/Nh0Bk3Y.png)  | As the preformance of all random adaptive sample algorithms proposed <br>preform similarly only Metropolis-Hastings proposals were adapted for <br>instance segmentation 

## Metropolis-Hastings
Below are the functions found within **Met_Hastings.py**

### MetHastingsBBox(img, ROI, pixels, bConst, roiConst, N)
**MetHastingsBBox** is the setup function to generate a feature map for the Metropolis-Hastings algorithm for a **Bounding 
Box** adaptation. The feature map function is found within distribution utils.
#### Inputs
* **img** - OpenCV input image (depth map)
* **ROI** - 2D numpy array with the x, y, width and height of the ROI stored respectively
* **pixels** - New amount of pixels to sub-sample the input image by
* **bConst** - background constant weight for feature map 
* **roiConst** - ROI constant weight for feature map (detected objects) 
* **N** - Iterations to provide to Met Hastings algorithm for number of proposal points for each new index found
#### Outputs
* **MH** - The new sub-sampled output (not-interpolated)

### MetHastingsInstance(img, mask, pixels, bConst, roiConst, N)
**MetHastingsBBox** is the setup function to generate a feature map for the Metropolis-Hastings algorithm for an **Instance 
Segmentation** adaptation. The feature map function is found within distribution utils.
#### Inputs
* **img** - OpenCV input image (depth map)
* **mask** - A mask defined by Mask-RCNN defining areas either true of false depending if an object is detected within that pixel or not
* **pixels** - New amount of pixels to sub-sample the input image by
* **bConst** - background constant weight for feature map 
* **roiConst** - ROI constant weight for feature map (detected objects) 
* **N** - Iterations to provide to Met Hastings algorithm for number of proposal points for each new index found
#### Outputs
* **MH** - The new sub-sampled output (not-interpolated)

### MetHastings(img, pixels, fMap, N)
**MetHastings** is a function used to apply the Metropolis-Hastings Markov chain Monte Carlo method to obtain a guided sequence 
of random samples. This process randomly samples a set of pixels within a scene and preforms an **N** 
number of proposal areas to change the initial randomly chosen index to a more favourable placement, based upon a weighted 
feature map.

The following psuedo code outlines the execution of this method:

* Create a feature map of weighting ROI and background appropriately
* Loop for a number of pixels
    * Randomly choose x and y indexes
    * **n** = image(x, y);
    * Loop **N** number of iterations
      * Randomly choose proposal x’ and y’ indexes
      * **n'** = image(x’, y’);
      * **α** = featuremap(**n’**)/featuremap(**n**)
      * sample r uniformly between 0 and 1
      * r is a uniform number between 0 and 1
      * If r < **α** accept this new proposal<br>
        (**n’** = **n**)

Where: <br>
* **N** is the number of proposals to investigate for each uniformly sampled point<br>
* **n** is the previous point<br>
* **n’** is the new point proposal<br>
* **α** is the ration between the new and old point within the feature map<br>
* **σ** is the variance of the new proposals from the original position.


#### Inputs
* **img** - OpenCV input image (depth map)
* **pixels** - New amount of pixels to sub-sample the input image by
* **fMap** - Feature map which has been pre-weighted
* **N** - Iterations to provide to Metropolis-Hastings algorithm for number of proposal points for each individual index
#### Outputs
* **AS** - The new sub-sampled output (not-interpolated)
#### Example

Bounding Box            |  Instance Segmentation
:-------------------------:|:-------------------------:
![](https://i.imgur.com/XxyqYDi.png)  |  ![](https://i.imgur.com/RaARUA0.png)

### RandomWalkMetHastingsBBox(img, ROI, pixels, bConst, roiConst, sigma, N)
**RandomWalkMetHastingsBBox** is the setup function to generate a feature map and a uniform spread of pixels for the 
Random Walk Metropolis-Hastings algorithm for an **Bounding Box** adaptation. The feature map and uniform spread function 
is found within distribution utils.
#### Inputs
* **img** - OpenCV input image (depth map)
* **ROI** - 2D numpy array with the x, y, width and height of the ROI stored respectively
* **pixels** - New amount of pixels to sub-sample the input image by
* **bConst** - background constant weight for feature map 
* **roiConst** - ROI constant weight for feature map (detected objects)
* **sigma** - The variance of the new proposals from the orginal position (The amount of 'drift' points have)
* **N** - Iterations to provide to Radnom Walk Met Hastings algorithm for number of proposal points for each individual index
#### Outputs
* **RWMH** - The new sub-sampled output (not-interpolated)

### RandomWalkMetHastingsInstance(img, mask, pixels, bConst, roiConst, sigma, N)
**RandomWalkMetHastingsInstance** is the setup function to generate a feature map and a uniform spread of pixels for the 
Random Walk Metropolis-Hastings algorithm for an **Instance segmentation** adaptation. The feature map and uniform spread 
function is found within distribution utils.
#### Inputs
* **img** - OpenCV input image (depth map)
* **mask** - A mask defined by Mask-RCNN defining areas either true of false depending if an object is detected within that pixel or not
* **pixels** - New amount of pixels to sub-sample the input image by
* **bConst** - background constant weight for feature map 
* **roiConst** - ROI constant weight for feature map (detected objects)
* **sigma** - The variance of the new proposals from the original position (The amount of 'drift' points have)
* **N** - Iterations to provide to Random Walk Met Hastings algorithm for number of proposal points for each individual index
#### Outputs
* **RWMH** - The new sub-sampled output (not-interpolated)

### RandomWalkMetHastings(img, AS, fMap, sigma, N)
**RandomWalkMetHastings**  implements the adaptation of the statistical distribution Metropolis-Hastings dubbed, Random 
Walk Metropolis-Hastings. This process is where the scene is uniformly sampled, and proposal points are compared against 
the previous proposal points on the feature map, which is weighted appropriately and either accepted or rejected, depending 
on a random variable. This results in the uniformly sampled points being 'drifted' towards regions of interest. The 
'drifting' of these points are dependant on a variance which is defined in the function inputs.

The following psuedo code outlines the execution of this method:
* Uniformly sample the scene
* Create a feature map of weighting ROI and background appropriately.
* Start a loop through all uniformly sampled points.
    * Loop **N** number of iterations
        * New proposal for this uniformly sampled point is:<br>
          **n’** = **n** + round(**σ***randn(1,1))
        * **α** = featuremap(**n’**)/featuremap(**n**)
	    * r is a uniform number between 0 and 1
	    * If r < **α** accept this new proposal <br>
        (**n’** = **n**)<br>

Where: <br>
* **N** is the number of proposals to investigate for each uniformly sampled point<br>
* **n** is the previous point<br>
* **n’** is the new point proposal<br>
* **α** is the ration between the new and old point within the feature map<br>
* **σ** is the variance of the new proposals from the original position.


#### Inputs
* **img** - OpenCV input image (depth map)
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
### walkIndex(prevI, max, sigma)
**walkIndex** generates an index proposal for the Random Walk Metropolis-Hastings algortihm, based upon the variance provided. 
If the variance is such that it may take the propsal index outwith the area of the image then the area which the proposal can 
be moved is adapted to ignore the points outwith the image.

#### Inputs
* **prevI** - The previous index to move from by the variance
* **max** - The maximum index allowed to ensure the proposal index is not out of range
* **sigma** - The varaince for the proposal from the original index point
#### Outputs
* **propI** - The new proposal index

### createFeatureMapBBox(img, ROI, bConst, rConst)
**createFeatureMapBBox** is the function to generate a feature map for a **Bounding Box** implementation of the model, The weights and locations of the ROI are provided in the inputs for the function.
#### Inputs 
* **img** - OpenCV input image (depth map)
* **ROI** - 2D numpy array with the x, y, width and height of the ROI stored respectively
* **bConst** - The weighting constant for the background
* **rConst** - The weighting constant for ROI 
#### Outputs
* **fMap** - An appropriately weighted feature map to be used in Metropolis-Hastings algorithms.

### createFeatureMapInstance(mask, bConst, iConst)
**createFeatureMapBBox** is the function to generate a feature map for a **Instance Segmentation** implementation of the model, The weights and mask are provided in the inputs for the function.
#### Inputs 
* **mask** - A mask defined by Mask-RCNN defining areas either true of false depending if an object is detected within that pixel or not
* **bConst** - The weighting constant for the background
* **rConst** - The weighting constant for ROI 
#### Outputs
* **fMap** - An appropriately weighted feature map to be used in Metropolis-Hastings algorithms.

### rmse(predictions, targets)
**rmse** is used to calculate the rmse between to variables
#### Inputs 
* **predictions** - Predicitons to be compared
* **targets** - Ground truth to compare predictions against
#### Outputs
* **RMSE** - The RMSE between the two variables

### seperateArrayInt(array, pixels)
**seperateArrayInt** function seperates a numpy array for nearest interpolation across the NaN points within the array.
#### Inputs
* **array** - The array to be interpolated
* **pixels** - The number of non-NaN instances within the array
#### Outputs
* **values** - The z value corresponding to x and y values within the points array
* **points** - The x and y values to the corresponding values within the values array

### seperateArrayPC(array, pixels)
**seperateArrayPC** function separates a numpy array for the x, y and z values to be saved into a .mat file for further processing.
#### Inputs
* **array** - The array to be interpolated
* **pixels** - The number of non-NaN instances within the array
#### Outputs
* **x** - The x values corresponding with the same indexes in the other arrays
* **y** - The arrays y values
* **z** - The arrays z values

### nInterp2D(pixels, array)
**nInterp2D** preforms nearest interpolation on a 2D numpy array.
#### Inputs
* **pixels** - The number of non-NaN instances within the array
* **array** - The array to be interpolated
#### Outputs
* **Nearest** - The newly interpolated array

### getNewDimensions(nPixels, oWidth, oHeight)
**getNewDimensions** gets calculates a new width and height, used to uniformly sub-sample the scene. Using the original 
width and height of the image and from the new number of allocated pixels a new height and width can be 
achieved. This approach was chosen as the aspect ratio for the area would stay the same regardless of the number of points 
to sample. 
#### Inputs
* **nPixels** - Points to sub-sample the area of the image to
* **oWidth** - Original width of the image
* **oHeight** - Original height of the image
#### Outputs
* **nWidth** - New sub-sampled width
* **nHeight** - New sub-sampled height

###checkRounding(limit, w, h)
**checkRounding** makes sure the rounding for an index does not excede the limit provided, used to prevent out of range 
exceptions due to rounding errors. 
#### Inputs
* **limit** - The limit for the indexes
* **w** - Width to be checked
* **h** - Height to be checked
#### Outputs
* **w** - Checked width
* **h** - Checked height

### uniformSpread(oArray, nPoints, nArray)
**uniformSpread** function uniformly sub-samples an array by a number of given points.
#### Inputs
* **oArray** - Original array to be sub-sampled
* **nPoints** - Number of points to sub-sample the input array by
* **nArray** - New array to be filled with sub-sampled points
#### Outputs
* **nArray** - Uniformly sub-sampled array

### nonNan(array)
**nonNan** checks an array and returns how many non-NaN values are within that array
#### Inputs
* **array** - Checks how many non-NaN values are within an array
#### Outputs
* **counter** - The number of non-Nan values in the array

### invertGrayscale(img)
**invertGrayscale** inverts a grayscale image
#### Inputs
* **img** - The grayscale image to be inverted
#### Outputs
* **img** - Grayscale converted image

### combineMasks(masks)
**combineMasks** flattens an array of masks into a single mask array.
#### Inputs
* **masks** - The mask array to be flattened
#### Outputs
* **mask** - Flattened mask array