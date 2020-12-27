## Writeup Template

---

**Advanced Lane Finding Project**

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

[image1]: ./my_examples/calibration_undistorted.png     "Undistorted"
[image2]: ./my_examples/original_vs_undistorted.png     "Road undistorted"
[image3]: ./my_examples/overlayed_undistorted.png       "Road overlayed"
[image4]: ./my_examples/filtered_image.png              "Filtered Example"
[image5]: ./my_examples/original_vs_warped.png          "Original vs warped"
[image6]: ./my_examples/radius_calculation.png          "Radius calculation"
[image7]: ./my_examples/pipeline_frame_1.jpg            "Pipeline frame"
[video1]: ./output_videos/project_video_out.mp4         "Output video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

First step in camera calibration was to map image points to the objects points of the chessboard image. For that I used all the sample images from the calibration folder and the procedure described in the materials.

Second step was to calculate calibration coefficients by calling openCV function:

    def undistort(src, cameraMatrix, distCoeffs, dst=None, newCameraMatrix=None)

![alt text][image1]

More details can be found in the main code file P2.ipynb

### Pipeline (single images)

Original vs undistorted:

![alt text][image2]

Original vs undistorted overlayed:

![alt text][image3]


#### Example of filtered image

I used a variation of the algorithm from the materials to create filtered binary image of the road lanes. Primary source for the filter is the HLS color space image transformation, where I combined Sobel gradients along x axis for both the L and S channel. This way I removed unnecessary data from the S channel in the same manner as for L channel. I had other ideas like using rotated filters ie:
```
K = [-2 -1 0
    -1  0 1
     0  1 2]
```
To better accentuate gradients over the lane lines, but due to lack of time didn't implement it. The implementation of the filter procedure is available in the procedure:

    def filter_image(img, l_thresh=(20, 100), s_thresh=(50, 120)):

Example of filtered and undistorted image frame:

![alt text][image4]

#### Perspective transform

For perspective transform I first tried to identify points on the image that should correspond to points in the real-world that form a straight rectangle. For that purpose I selected a frame of the video with the straight section of the road, and selected point along the lane lines (since lines should be parallel in these conditions). I selected these points:

```
# Assumed rectangle corner points from the image
# down left, up left, up right, down right
pts = np.array([[312, 651], [584, 454], [698, 454], [1024, 651]], np.int32)
```

I then set a destination rectangle so that height of the source rectangle corresponds to the full image height. Bottom of the destination rectangle is scaled to take right amount of space on the sides of the lane to be able to track the curvature, but not more then necessary to avoid other objects and lines.

```
x_length = np.int32((1024 - 312) * 0.9)
x_left = (1280 - x_length) / 2
x_right = 1280 - xleft

dst = np.float32([[x_left, 720], [x_left, 0], [x_right, 0], [x_right, 720]])
```

Example of frame that is warped to bird-eye perspective using previous data:

![alt_text][image5]

#### Identification of lane-lines

As was suggested during the course, I used two methods for identifying lane-line pixels:

1. The "mirror" algorithm
2. Selecting pixels in the vicinity of previosly fitted lines

Mirror algorithm is implemented in the method:

    def find_pixels_mirror(binary_warped, params):

This method uses a histogram of the number of detected points for a given x position, then finding the x position with the highest number of pixels as starting pointa for detection. Afterwards, the detection is performed by calculating the number of pixels within a mirror of predefined size. If the number of points is larger than some predefined value, the x-position average of these pixels becomes the next center point for following mirrors. The set of all detected points is used for polynomial fitting of the lane-lines. This method is normaly used as starting step because of its robustness.

The second algorithm uses previously fitted lane-lines to determine the set of pixels used for the next estimation. Hence it is only used after some initial estimation has already been done, for example by the mirror algorithm. It is probably the more efficient algorithm of the two. This algorithm is implemented in:

    def find_pixels_poly(binary_warped, left_fit, right_fit, params):

#### Calculation of radius and offset from the middle of the lane

Radius of the lane-lines was calculating by using the formula:

![][image6]

from the materials. Basically for any curve that is at least two times differentiable and the second derivative is not zero at particular point, we can calculate the curvature radius at that point using the formulas above.

For my pipeline I have estimated the curvature based on the description in the materials, using the averaged data from 4th to 7th second of the video. In this part of the video the curvature is pretty consistent so it seemed as a good source for calculating the radius. The estimated value is:
```
    Averaged lane curve: 1270.033268 m
```

One particular detail from the implementation is that I transformed the coefficients of polynomial for lane-line in pixel space to line-lines in "real" space by using the formulae:
```
    left_fit_m = left_fit * [xm_per_pix/ym_per_pix**2, xm_per_pix/ym_per_pix, xm_per_pix]
    right_fit_m = right_fit * [xm_per_pix/ym_per_pix**2, xm_per_pix/ym_per_pix, xm_per_pix]
```
It is simply obtained by replacement in the original equation {y_pix -> y_m * ym_per_pix}, and similarly for the x value. The method for radius estimation is:

    def radius_measurements(left_fit, right_fit, lane_params):

Middle of the lane was calculated by finding the middle from the starting points of the lane-lines (with the highest y_m, corresponding to highest y_pix which is 658 for my region). Difference between the middle of the image (calculated from the right, in [m]) and this value is the estimate of the ofsset in [m].

A note, in my pipeline the region is roughly similar to the region from the materials, so the ym_per_pix and xm_per_pix should still be applicable (roughly).

Method for calculating the offset:

    def position_measurement(left_fit, right_fit, lane_params):

Here is one frame from the pipeline output video that shows the detected lane-lines with offset from the middle overlay:

![][image7]

---

### Pipeline (video)

Here's a [link to final output video][video1].


---

### Discussion

Probably the most important step in the algorithm for finding lane-lines is to have good binary data when filtering the image. It is much more difficult to compensate the lack of good binary data later in the algorithm. I tried to use separate filtering for the left and right lane-lines, by using specific rotated filters. However, there were additional complexities to overcome so I didn't pursue it further.

For increasing the stability of the estimations I used simple averaging of previous detections, but some more robust statistical method would be better suited for the task. For example estimations that are to far off would be better ignored completely and not used for averaging.

The averaging of the previous estimations made detection less sensitive, which can be seen for the part of the road where the car bouncing is more pronounced. In such cases the estimation lags behind the current lane positions, which may or may not be important?



