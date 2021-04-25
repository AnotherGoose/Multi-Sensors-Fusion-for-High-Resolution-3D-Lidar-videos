# Guided Filtering
Guided Filtering is one of three proposed method used for Denoisng 3D LIDAR data. 

Guided Filtering is a filtering technique which uses a guidance image (I) to improve the clarity of the input image. Guided Filtering is quite versatile since as an edge preserving smoothing process it does not suffer from artifacts near the edges. The input image is a set of data (depth map, point cloud, reflective map etc.) of a target object which requires some altercations because it's affected by glaring issues (such as ambient noise or low scale resolution). A guidance image is usually a more concise data set of the target object, which helps direct the input image into becoming a refurbished version of the target object. 

For this use case, Guided Filtering is used to denoise a video provided by (one of two) Adaptive sampling techniques frame by frame then accumulate the denoised frames into a point cloud to replicate a video playing.

## Requirements
All code for this denoiser was made on MATLAB R2020a and using the following toolboxes:

UAV toolbox: <https://uk.mathworks.com/products/uav.html>
LIDAR toolbox: <https://uk.mathworks.com/products/lidar.html>
Computer Vision toolbox: <https://uk.mathworks.com/products/computer-vision.html>


## Acknowledgements
This denoiser is has been adapted for 3D functionality using structured and unstructured data inputs. This verison of Guided Filtering as been adpated from He Kaiming (et all.).

Original Guided Filtering .mat files: <https://github.com/accessify/fast-guided-filter>











