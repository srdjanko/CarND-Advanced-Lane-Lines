import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging

def draw_polyline(img, vertices, color=[255, 0, 0], thickness=2, Closed = False):
    """
    Simple method for drawing connected lines or polygons, given the
    set of points. Starting and ending point can be connected automatically
    to form closed polygon figure.
    """
    cv2.polylines(img, vertices, Closed, color, thickness, lineType=cv2.LINE_AA)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Simple method for drawing set of individual lines, each defined by start and
    end points.
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_label(img, text, pos, scale = 0.7, color = (0,0,0)):
    """
    Method for displaying text on given part of the image.
    """
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def filter_image(img, l_thresh=(20, 100), s_thresh=(50, 120)):
    """
    Taken from materials and modified.

    Performs image filtering based on the L and S channel (HLS), where each
    channel is filtered separately, thresholded, and then combined into single
    binary output. This is different from the material version where the S binary
    is directly combined with the sobel binary.
    """
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x, l channel
    sobel_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobel_l = np.absolute(sobel_l) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_l = np.uint8(255*abs_sobel_l/np.max(abs_sobel_l))

    # Threshold x gradient
    l_binary = np.zeros_like(scaled_sobel_l)
    l_binary[(scaled_sobel_l >= l_thresh[0]) & (scaled_sobel_l <= l_thresh[1])] = 1

    # Sobel x, s channel
    sobel_s = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobel_s = np.absolute(sobel_s) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_s = np.uint8(255*abs_sobel_s/np.max(abs_sobel_s))

    # Threshold x gradient
    s_binary = np.zeros_like(scaled_sobel_s)
    s_binary[(scaled_sobel_s >= s_thresh[0]) & (scaled_sobel_s <= s_thresh[1])] = 1

    s_l_binary = np.zeros_like(s_binary)
    s_l_binary[(s_binary == 1) | (l_binary == 1)] = 1

    return s_l_binary

def find_pixels_mirror(binary_warped, params):
    """
    Taken from materials.

    Method performs inital lane detection by using the 'mirror' algorithm. Since
    this method is likely resource intensive, it is only used in initial estimation
    or as fallback when other methods fail.
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = params['nwindows']  #9
    # Set the width of the windows +/- margin
    margin = params['margin'] # 100
    # Set minimum number of pixels found to recenter window
    minpix = params['minpix'] # 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def find_pixels_poly(binary_warped, left_fit, right_fit, params):
    """
    Taken from materials and tailored for this pipeline.

    Based on existing lane polynomials, we choose a region with given margin arround
    the polynomial curves. This region is then used to select candidate points for next
    estimation. This algorithm is likely faster then 'mirror' algorithm is used whenever
    possible.
    """
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = params['margin'] # 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def plot_debug(binary_warped, left_x_nonzero, left_y_nonzero, right_x_nonzero, right_y_nonzero,
                left_fit_poly, right_fit_poly, margin):

    """
    Taken from materials.

    Visualization of relevant debug information, to better estimate the quality of the
    lane detection pipeline.
    """
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[left_y_nonzero, left_x_nonzero] = [255, 0, 0]
    out_img[right_y_nonzero, right_x_nonzero] = [0, 0, 255]

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    try:
        left_fitx = left_fit_poly[0]*ploty**2 + left_fit_poly[1]*ploty + left_fit_poly[2]
        right_fitx = right_fit_poly[0]*ploty**2 + right_fit_poly[1]*ploty + right_fit_poly[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    draw_left = (np.asarray([left_fitx, ploty]).T).astype(np.int32)
    draw_right = (np.asarray([right_fitx, ploty]).T).astype(np.int32)

    cv2.polylines(result, [draw_left], False, (255,0,0), thickness=5)
    cv2.polylines(result, [draw_right], False, (255,0,0), thickness=5)

    return result

def plot_lanes(undist, Minv, left_fit_poly, right_fit_poly):
    """
    Taken from materials.

    Final visualization of the lane lines.
    """
    # Generate x and y values for plotting
    img_shape = undist.shape

    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0] )
    try:
        left_fitx = left_fit_poly[0]*ploty**2 + left_fit_poly[1]*ploty + left_fit_poly[2]
        right_fitx = right_fit_poly[0]*ploty**2 + right_fit_poly[1]*ploty + right_fit_poly[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Create an image to draw the lines on
    warp_zero = np.zeros(img_shape[:2]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    draw_left = (np.asarray([left_fitx, ploty]).T).astype(np.int32)
    draw_right = (np.asarray([right_fitx, ploty]).T).astype(np.int32)
    cv2.polylines(color_warp, [draw_left], False, (255,0,0), thickness=5)
    cv2.polylines(color_warp, [draw_right], False, (255,0,0), thickness=5)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

    return result

def plot_poly(ploty, poly):
    """
    Taken from the materials and modified.

    Returns a set of plotx points calulated from the polynomial and input ploty data.
    """

    fit_success = False

    try:
        plotx = poly[0]*ploty**2 + poly[1]*ploty + poly[2]
        fit_success = True
    except TypeError:
        # Avoids an error if poly is still none or incorrect
        print('The function failed to fit a line!')
        plotx = 1*ploty**2 + 1*ploty

    return plotx, fit_success

def fit_poly_to_points(x, y):
    """
    Taken from the materials.

    Based on the detected points, calculate polynomials of the lane curve.
    """
    fit_success = True

    try:
        fit = np.polyfit(x, y, 2)
    except np.RankWarning:
        # In case if polyfit fails, return coefficients of x = 0 line
        fit = [0, 0, 0]
        fit_success = False

    return fit, fit_success

def fit_poly_to_lanes(warped_binary):
    """
    Procedure for detecting road lanes based on the binary pixel data, obtained by filtering and
    warping each recorded frame.
    """

    import globals
    lane_params = globals.lane_params

    # Fetch previously detected lanes
    lanes = lane_params.detected_lanes

    # Current lane
    current_lane = globals.Lane_fits()
    lanes_length = len(lanes)

    if lanes_length == 0:
        # Try new mirror detection sequence
        leftx, lefty, rightx, righty = find_pixels_mirror(warped_binary, lane_params.find_pixels_mirror)

    else:
        # Use previous best fit to define fit area
        average_lane = lane_params.best_fit
        leftx, lefty, rightx, righty = find_pixels_poly(warped_binary, average_lane.left_fit,
                                                        average_lane.right_fit, lane_params.find_pixels_poly)

    # Calculate polynomial from detected points
    left_fit, left_fit_success = fit_poly_to_points(lefty, leftx)
    right_fit, right_fit_success = fit_poly_to_points(righty, rightx)
    fit_success = left_fit_success & right_fit_success

    current_lane.left_fit = left_fit
    current_lane.right_fit = right_fit
    current_lane.fit_success = fit_success

    if (not fit_success) and (lanes_length == 0):
        logging.warning('Lane detection not successful.')

    if current_lane.fit_success:
        lanes.insert(0, current_lane)

    # Best fit
    best_fit = globals.find_lane_average(lanes)
    lane_params.best_fit = best_fit

    if len(lanes) > lane_params.lane_count:
        lanes.pop()

    return leftx, lefty, rightx, righty, best_fit

def radius_measurements(left_fit, right_fit, lane_params):
    '''
    Taken from the materials and adapted for the pipeline.

    Calculates the radius of the curvature of the lanes in [m].
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image

    # Lambda for calculating curvature radius in pixels
    # Comvert from lines measured in pixels to lines measured in meters
    xm_per_pix = lane_params.xm_per_pix
    ym_per_pix = lane_params.ym_per_pix

    # The bottom left point of the region, at the same time the lowest point of the curves
    y_pix = lane_params.lane_region[0][1]
    y_m = y_pix * ym_per_pix

    left_fit_m = left_fit * [xm_per_pix/ym_per_pix**2, xm_per_pix/ym_per_pix, xm_per_pix]
    right_fit_m = right_fit * [xm_per_pix/ym_per_pix**2, xm_per_pix/ym_per_pix, xm_per_pix]

    curv = lambda a, b, y : (1 + (2*a*y + b)**2)**(1.5) / np.abs(2*a)

    left_curverad = curv(left_fit_m[0], left_fit_m[1], y_m)
    right_curverad = curv(right_fit_m[0], right_fit_m[1], y_m)

    return left_curverad, right_curverad

def position_measurement(left_fit, right_fit, lane_params):
    '''
    Taken from the materials and adapted for the pipeline.

    Calculates the vehicle offset from the middle of the lane in [m].
    '''
    # Comvert from lines measured in pixels to lines measured in meters
    xm_per_pix = lane_params.xm_per_pix
    ym_per_pix = lane_params.ym_per_pix

    # The bottom left point of the region, at the same time the lowest point of the curves
    y_pix = lane_params.lane_region[0][1]
    y_m = y_pix * ym_per_pix

    left_fit_m = left_fit * lane_params.transform_poly_2_m
    right_fit_m = right_fit * lane_params.transform_poly_2_m

    # Calculate position from middle of the lane
    left_curve_pos = left_fit_m[0]*y_m**2 + left_fit_m[1]*y_m + left_fit_m[2]
    right_curve_pos = right_fit_m[0]*y_m**2 + right_fit_m[1]*y_m + right_fit_m[2]

    lane_middle_pos = (left_curve_pos + right_curve_pos) / 2
    image_middle_pos = lane_params.img_shape[1] * xm_per_pix / 2

    # Since x values grow to the right, positive values here mean vehicle is shifted to
    # the right of the lane middle
    vehicle_pos = image_middle_pos - lane_middle_pos

    return vehicle_pos