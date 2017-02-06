# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, my goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4).

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



## Dependencies
This project requires Python 3.5 and the following Python libraries installed:
* numpy
* opencv
* moviepy
* skimage
* sklearn


Implementation details
---
### 1. Data Preprocessing

The easiest way to do this is to investigate an image where the lane lines are straight, and find four points lying along the lines that, after perspective transform, make the lines look straight and vertical from a bird's eye view perspective(top down view).

### 2. Feature Extraction
Three types of features (2628 features in total) are used for the vehicle detection:
- Spatial features
- Color histogram features (RGB histogram and HLS histogram)
- Histogram of gradient (HOG) features


Specifically, The spatial feature uses the raw pixel values of the images and flattens them into a vector.  The Color Histogram remove the structural relation and allow more flexibility to the variance of the image, which allow us to identify the different appearance of the car.  The Histogram of Gradient Orientation (HOG) is also used to capture the signature for a shape and allows variation.


### 3. Build SVM Classifier

The SVM classifier was was chosen because it has a good balance of performance and speed.  It trained on 14208 images, and tested on 3552 images with 0.964 accuracy on testing set.    

### 4. Vehicle Detection
Then following techniques are used to identify and determine car part from non-car part in the image.

1. Sliding Window Search
2. Extract Window Features  
3. Create Heatmap
4. Remove Duplicates and Find Center of Car
5. Estimate Bounding box




## Data
Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  


## Reflection
1. Thresholds for heat-map are hard coded arbitrary, which can be improved by assigning dynamic values by learning the image/video in real time
2. The pipeline does not work well for more complicated images with cases.   Adding smoothing methods may be helpful in increasing accuracy in predicting shading cases and reduce false positive detection.
3. More real-time information may be added for analysis/diagnosis purpose.  Potentially, we could combine the lane finding methods with vehicle detection methods in real-time vehicle monitoring.



Reference
---
- http://scikit-image.org/docs/dev/auto_examples/plot_hog.html
- https://github.com/GeoffBreemer/SDC-Term1-P5-Vehicle-Detection-and-Tracking
- https://github.com/Dalaska/CarND-P5-Vehicle-Detection-and-Tracking
