# Task-Based Adaptive Sampling
Task Based Adaptive Sampling is proposed method within the topic of Adaptive sampling for a Lidar. 

Task based Task-Based Adaptive Sampling is where a task dictates the sampling of the Lidar. A task, in this case, is an object determined via a pre-trained image recognition algorithm. Objects identified within the scene will define regions of interests (ROI’s). ROI’s are a concept that propose an area to perform further processing within the scene.  

In the case of Task-Based Adaptive Sampling, objects detected provide ROI’s which are scanned at a higher rate than areas out with the ROI’s, providing a more focussed sampling of the scene compared to unform or random sampling. 

## Image Recognition Methods
This adaptation of task based adaptive sampling used the pre-trained networks and provided weights from the following repositories 
* Bounding Box - **YOLO** (<https://github.com/AlexeyAB/darknet>)
* Instance Segmentation - **Mask-RCNN** (<https://github.com/matterport/Mask_RCNN>)
## Getting Started
It is highly recommended that you use Anaconda environments for the install and execution of this implementation for Task-Based Adaptive Sampling

### Prerequisites

What things you need to install the software and how to install them

- Python 3.6.13
- Tensorflow 1.14.0
- Keras 2.3.0
- OpenCV-Python Latest Version
- h5py 2.10.0
- imgaug Latest Version
- mask_RCNN_coco.h5
- pycocotools

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```
### Execution


## Pixel Distribution

## Acknowledgements
### YOLO
This implementation for YOLO was 

<Mhttps://www.youtube.com/watch?v=FE2GBeKuqpc&t=296s&ab_channel=TheCodingBug>

### Mask - RCNN
The following youtube videos were extremly helpful in the building and installing for Mask-RCNN

<https://www.youtube.com/watch?v=2TikTv6PWDw&ab_channel=MarkJay>
<https://www.youtube.com/watch?v=epTfYW6oDqA&ab_channel=MachineLearningHub>

