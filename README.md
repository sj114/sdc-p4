##README
###Advanced Lane Finding Project

---

**Objectives**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image_cal_orig]: ./output_images/calibration3.jpg "Original Calibration Image"
[image1]: ./output_images/calibration3_output.jpg "Identified Chessboard Corners"
[image2]: ./output_images/calibration3_undist.jpg "Undistorted Chessboard"
[image_road_undist]: ./output_images/road_undist.jpg "Road Transformed"
[image_binary]: ./output_images/color_binary.png "Binary Example"
[image_warped]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image_output]: ./output_images/output_solidWhiteRight.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 54 through 79 of `camera_cal.py`.  

The object points are (x, y, z) coordinates of the chessboard corners in the world, with the assumption that z=0. The object points are the same for each calibration image. `objpoints` will be appended with a copy of these coordinates every time chessboard corners are successfully detected in a test image.  `imgpoints` contain the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

`objpoints` and `imgpoints` are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The calibration data is saved into a pickle file that will be subsequently used in the pipeline to undistort the road images.

![alt text][image_cal_orig]
![alt text][image1]
![alt text][image2]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The distortion coefficients and camera calibration matrix are loaded from the saved pickle file and applied to every  image in the pipeline. Here's an example:

![alt text][image_road_undist]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The `detect_edges` function in pipeline.py converts the input BGR image to HLS color space. The L and S channels are extracted since they contain the most relevant lighting and saturation features that aid in line/edge detection. 

The Sobel function is used to compute the gradient of the image in the x and y directions and they are then thresholded and re-stacked to create a binary image.

![alt text][image_binary]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_perspective_transform()` and `four_point_transform`, which appears in lines 1 through 8 in the file `pipeline.py`. I chose the source and destination points in the following manner:

```
pts = np.array([(625, img.shape[0]-300),
                (662, img.shape[0]-300),
                (1125,img.shape[0]),
                (200, img.shape[0])], dtype = "float32") 
dst = np.array([
            [img.shape[1]/4-20, bl[1]/6], 
            [img.shape[1]*3/4-10, br[1]/6], 
            [img.shape[1]*3/4-10, br[1]], 
            [img.shape[1]/4-20, bl[1]]], dtype = "float32")

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 530, 486      | 300, 120      | 
| 764, 486      | 950, 120      |
| 1125, 720     | 950, 720      |
| 200, 720      | 300, 720      |

This perspective transform was verified by drawing the `src` and `dst` points onto a test image, and by checking that the lines appear parallel in the warped image.

![alt text][image_warped]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Blind search:
* When starting from scratch, a histogram-based search is performed on the warped and thresholded image. 
* The histogram is divided into two halves (left and right) and the peaks are identified in each half. The x-coord of the peak is taken to be the starting point to search for the lane line pixels. 
* A sliding window of 48x48 pixels is applied around the above-computed starting point and all non-zero pixels are assumed to be potential lane line points.
* The histogram and peak calculations proceed by sliding the histogram window up in every iteration till the top of the warped image is reached, and the sliding window algorithm is applied to accrue potential lane line points.
* np.polyfit is then used to find a second order polynomial to fit the points identified in the above algorithm.

Here is an example of the fitted lane line:

![alt text][image5]

Intelligent search:
* In a video stream, the lane line estimation from the previous frame can be used to aid in the detection for the current frame, since the lane location can not change significantly between two consecutive frames.
* Instead of taking histogram of the warped images, this approach uses the fitted lane lines/curves from the previous frame and uses the same sliding window algorithm with a window of 48x48 along the previous frame's fitted curve to search for the current frame's lane line pixels. 
* np.polyfit is then used to find a second order polynomial to fit the new points identified.

Smoothing:
* In order to smooth out the detection, the lane line points and polynomial fit coefficients are averaged over the previous 10 frames.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

The radius of curvature has been estimated as described in the lecture notes. 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This has been implemented in `pipeline.py` in the function `draw_lanes()` (~line 337-355) by using cv2.fillPoly() and applying the inverse warp transform through cv2.warpPerspective(). The filled polygon image is superimposed onto the original image to get the final output image. 

Here is an example of the result on a test image:

![alt text][image_output]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

