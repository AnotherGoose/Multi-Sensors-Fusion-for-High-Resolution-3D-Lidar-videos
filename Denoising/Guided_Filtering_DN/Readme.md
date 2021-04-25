# Guided Filtering
Guided Filtering is one of three proposed method used for Denoisng 3D LIDAR data. 

Guided Filtering is a filtering technique which uses a guidance image (I) to improve the clarity of the input image. Guided Filtering is quite versatile since as an edge preserving smoothing process it does not suffer from artifacts near the edges. The input image is a set of data (depth map, point cloud, reflective map etc.) of a target object which requires some altercations because it's affected by glaring issues (such as ambient noise or low scale resolution). A guidance image is usually a more concise data set of the target object, which helps direct the input image into becoming a refurbished version of the target object. 
For this use case, Guided Filtering is used to denoise a video provided by (one of two) Adaptive sampling techniques frame by frame.






