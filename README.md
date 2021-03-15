# Multi-Sensors-Fusion-for-High-Resolution-3D-Lidar-videos

This repository is the demonstration of various improvements to a base implementation of Lidar video. 

This demonstration employs various techniques, such as:

* **Adaptive Sampling**
    * Task-Based Adaptive Sampling
    * Feature Extraction
* **Denoising**
    * RED
    * Guided Filtering
    * Inverse Problems
* **Super-Resolution**
    * Inverse Problems
    
This repository is a collaboration of work between:
* **Julia Bonat** - Feature Extraction
* **Ronan Kingsbury** - RED
* **Jack Maxwell** - Task-Based Adaptive Sampling
* **Saif Saleem** - Inverse Problems
* **Fezan Tabassum** - Guided Filtering

### Prerequisites

To download this repo, you will need to install git LFS: 
<https://git-lfs.github.com/>

## Sample Data
The input data for this demonstration is the dataset from the MPI Sintel Flow.
MPI Sintel is a dataset which is used for the evaluation of optical flow techniques, derived from an Open Source 3D animated short film, "Sintel"

<http://sintel.is.tue.mpg.de/>

This dataset is used as it provides depth video paired with RGB video along with distortion for the data. The pairing of RGB data with Depth information is useful for the techniques explored in this demonstration like Task-Based Adaptive Sampling or Guided Filtering
