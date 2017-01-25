import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Lane(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = deque() 
        self.recent_polyfits = deque() 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = np.linspace(0, 100, num=101)*7.2

    def _avg_n(self, a, item):
        a.append(item)
        
        # store only last 10 sets of data
        if len(a) > 10:
            a.popleft()

        # average the x values of the last n frames
        return a, np.mean(a, axis=0)

    def smooth_fit(self):
        self.recent_xfitted, self.bestx = self._avg_n(self.recent_xfitted, self.allx)
        self.recent_polyfits, self.best_fit = self._avg_n(self.recent_polyfits, self.current_fit)

    def compute_fit(self, lane_points, color='red'):
        # Fit a second order polynomial to each fake lane line
        _yvals = np.array(lane_points)[:,1]
        _xvals = np.array(lane_points)[:,0]
        self.current_fit = np.polyfit(_yvals, _xvals, 2)

        self.allx = self.current_fit[0]*self.ally**2 \
                        + self.current_fit[1]*self.ally \
                        + self.current_fit[2]

        self.detected = True  

        # Plot up the fake data
        #plt.plot(_xvals, _yvals, 'o', color=color)
        #plt.xlim(0, 1280)
        #plt.ylim(0, 720)
        #plt.plot(self.allx, self.ally, color='green', linewidth=3)
        #plt.gca().invert_yaxis() # to visualize as we do the images
        #plt.show()

    def get_x(self, y):
        x = self.current_fit[0]*y**2 + self.current_fit[1]*y + self.current_fit[2]
        return x

    def get_radius_curvature(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 20/720 # metres per pixel in y dimension
        xm_per_pix = 3.7/650 # metres per pixel in x dimension

        y_eval = np.max(self.ally)/2

        fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        # Now our radius of curvature is in meters
        #print(self.radius_of_curvature, 'm')

    def get_vehicle_position(self):
        xm_per_pix = 3.7/650 # metres per pixel in x dimension
        pixels_off_center = int(self.get_x(np.max(self.ally)) - (1280/2))
        self.line_base_pos = xm_per_pix * pixels_off_center
        return self.line_base_pos

