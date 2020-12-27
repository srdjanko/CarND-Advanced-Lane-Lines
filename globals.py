import numpy as np

class Lane_fits():
    def __init__(self):
        self.left_fit = np.zeros((3), dtype=float)
        self.right_fit = np.zeros((3), dtype=float)
        self.fit_success = False

class Line():
    def __init__(self):
        # Number of previously detected lanes we want to average over
        self.lane_count = 4
        # Lane coefficients over last n iterations
        self.detected_lanes = []
        # Best estimation for the current iteration
        self.best_fit = Lane_fits()
        # Pixel to length conversion parameters
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700
        self.transform_poly_2_m = [self.xm_per_pix/self.ym_per_pix**2,
            self.xm_per_pix/self.ym_per_pix, self.xm_per_pix]

        # Image filtering parameters
        self.l_thresh = (20, 100)
        self.s_thresh = (50, 120)

        # Lane detection parameters
        self.find_pixels_mirror = {'nwindows': 9, 'margin': 100, 'minpix': 50}
        self.find_pixels_poly = {'margin': 100}
        self.img_shape = (720, 1280)

        # Lane region: BL, UL, UR, BR
        self.lane_region = np.array([[309, 658], [580, 460], [706, 460], [1042, 658]], np.int32)

def find_lane_average(lanes):

    count = len(lanes)
    average = Lane_fits()
    for lane in lanes:
        average.left_fit += lane.left_fit / count
        average.right_fit += lane.right_fit / count

    return average

lane_params = Line()
