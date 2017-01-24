import pickle
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from IPython.display import HTML

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("camera_cal.pickle", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
M = []
Minv = []

class Lane(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
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

    def compute_fit(self, lane_points):
        # Fit a second order polynomial to each fake lane line
        _yvals = np.array(lane_points)[:,1]
        _xvals = np.array(lane_points)[:,0]
        self.current_fit = np.polyfit(_yvals, _xvals, 2)

        # Plot up the fake data
        #plt.plot(leftx, left_yvals, 'o', color='red')
        #plt.plot(rightx, right_yvals, 'o', color='blue')
        #plt.xlim(0, 1280)
        #plt.ylim(0, 720)
        self.allx = self.current_fit[0]*self.ally**2 \
                        + self.current_fit[1]*self.ally \
                        + self.current_fit[2]

        #plt.plot(left_fitx, left_yvals, color='green', linewidth=3)
        #plt.plot(right_fitx, right_yvals, color='green', linewidth=3)
        #plt.gca().invert_yaxis() # to visualize as we do the images
        #plt.show()

    def get_x(self, y):
        x = self.current_fit[0]*y**2 + self.current_fit[1]*y + self.current_fit[2]
        return x

    def get_radius_curvature(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        y_eval = np.max(self.ally)/2

        fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        # Now our radius of curvature is in meters
        #print(self.radius_of_curvature, 'm')

    def get_vehicle_position(self):
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        pixels_off_center = int(self.get_x(np.max(self.ally)) - (1280/2))
        self.line_base_pos = xm_per_pix * pixels_off_center
        return self.line_base_pos

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    print(rect)
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = pts
    (tl, tr, br, bl) = pts

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    offx = 320
    offy = 169
    print ("maxH: ", maxHeight)
    dst = np.array([
            [0 + offx, 0],
            [maxWidth - 1 + 40, 0],
            [maxWidth - 1 + 40, maxHeight - 1 + offy],
            [0 + offx, maxHeight - 1 + offy]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    # return the warped image
    return M, maxWidth, maxHeight

def my_four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = pts
    (tl, tr, br, bl) = pts

    # find the line passing through left points
    ml = (tl[1]-bl[1])/(tl[0]-bl[0])
    cl = bl[1] - ml*bl[0]

    # find the line passing through right points
    mr = (tr[1]-br[1])/(tr[0]-br[0])
    cr = br[1] - mr*br[0]

    # find the vanishing point
    vx = (cr - cl)/(ml - mr)
    vy = ml * vx + cl

    print ("Vanishing point: (", vx, ",", vy, ")")

    # compute the measuring points
    mpl_x = vx - (br[1] - vy)
    mpl_y = vy
    mpr_x = vx + (br[1] - vy)
    mpr_y = vy
    
    print ("Measuring points: ", mpl_x, mpl_y, mpr_x, mpr_y)

    # find the line passing through measuring point and bottom src points
    m1 = (mpr_y-bl[1])/(mpr_x-bl[0])
    c1 = bl[1] - m1*bl[0]
    m2 = (mpl_y-br[1])/(mpl_x-br[0])
    c2 = br[1] - m2*br[0]

    # find the intersection of the projected diagonal lines and the left and
    # right lane lines
    tl[0] = (c2 - cl)/(ml - m2)
    tl[1] = ml * tl[0] + cl
    tr[0] = (c1 - cr)/(mr - m1)
    tr[1] = mr * tr[0] + cr

    print ("New src points: ", tl[0], tl[1], tr[0], tr[1])

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
            [300, 119], [950, 119], [950, 719], [300, 719]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = cv2.getPerspectiveTransform(dst, rect)

    # return the warped image
    return M, Minv, maxWidth, maxHeight

def detect_edges(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    undist = cv2.undistort(img, mtx, dist, None, mtx)    
    img = np.copy(undist)
    
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    img_size = color_binary.shape[::-1]
    #print ("Full image size: ", img_size)
    return color_binary

def get_perspective_transform(img):
   # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # plt.imshow(img, cmap='binary')
   # plt.grid(True)
   # plt.grid(b=True, which='major', color='b', linestyle='-')
   # plt.grid(b=True, which='minor', color='r', linestyle='--')
   # plt.minorticks_on()
   # plt.show()

    # this is on whole image of 1280x720 (tl, tr, br, bl)
    #pts = np.array([(600,440),(700,440),(960,620),(350,620)], dtype = "float32") #section of the lane
    pts = np.array([(625,420),(662,420),(1125,719),(200,719)], dtype = "float32") #full lane
     
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    M, Minv, maxWidth, maxHeight = my_four_point_transform(img, pts)
    return M, Minv

def do_birds_eye(img, M):
    warped = cv2.warpPerspective(img, M, (1280,720)) #650,330 if section of lane
    return warped
   
''' Create a sliding window and find all lane points within it
'''
def sliding_window(img, step_size, window_size, index, lane_points):
    window = img[img.shape[0]-(step_size+window_size):img.shape[0]-step_size, index - window_size/2:index + window_size/2]
    l_channel = window[:,:,1]
    s_channel = window[:,:,2]
    for i in range(window_size):
        for j in range(window_size):
            if window.shape[0] != window_size or window.shape[1] != window_size:
                print("Hey, we have a prob!", step_size, index)

            if(l_channel[i, j] != 0 or s_channel[i, j] != 0):
                x_coord = index - window_size/2 + j
                y_coord = img.shape[0]-(step_size+window_size) + i
                lane_points.append((x_coord,y_coord)) 
                #all_points.append((x_coord,y_coord)) 

    return lane_points

import peakutils
from peakutils.plot import plot as pplot
def blind_search(img):
    # histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    # splitting the above histogram into left and right sections
    hist_len_half = int(len(histogram)/2)
    hist_left = histogram[:,1][0:hist_len_half] + histogram[:,2][0:hist_len_half]
    hist_right = histogram[:,1][hist_len_half:len(histogram)] + histogram[:,2][hist_len_half:len(histogram)]
    #plt.plot(histogram[:,1])
    #plt.show()
    #print("Histogram shape: ", histogram.shape)
    #print("Max hist[0]:", max(histogram[:,0]))
    #print("Max hist[1]:", max(histogram[:,1]))
    #print("Max hist[2]:", max(histogram[:,2]))

    # Identify peaks in the histogram and get first estimate of lane start
    index_l = max(peakutils.indexes(hist_left, thres=0.5, min_dist=30))
    x = np.linspace(0, hist_len_half-1, hist_len_half)
    print("Left Peaks: ", index_l, ", Values at peak points: ", x[index_l], hist_left[index_l])
    index_r = max(peakutils.indexes(hist_right, thres=0.5, min_dist=30))
    print("Right Peaks: ", index_r, ", Values at peak points: ", x[index_r], hist_right[index_r])
    index_r += hist_len_half
#    plt.figure(figsize=(10,6))
#    pplot(x, histogram[:,1], indexes)
#    plt.title('First estimate')
#    plt.show()

    return index_l, index_r, hist_len_half

def blind_identify_lane_pixels(img, left_lane, right_lane):

    index_l, index_r, hist_len_half = blind_search(img)
    print ("Performing a blind search")

    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    hist_len_half = int(len(histogram)/2)

    # use sliding window to identify rest of the lane
    hist_step = 60 #pixels
    window_size = 48
    left_lane_points=[]
    right_lane_points=[]
    all_points=[]
    index_r_offset = hist_len_half
    index_l_offset = 0

    hist_left = histogram[:,1][index_l-hist_step:index_l+hist_step] + histogram[:,2][index_l-hist_step:index_l+hist_step]
    hist_right = histogram[:,1][index_r-hist_step:index_r+hist_step] + histogram[:,2][index_r-hist_step:index_r+hist_step]
    index_l_offset = index_l-hist_step
    index_r_offset = index_r-hist_step

    for step_size in range(0, img.shape[0], window_size):
        try:
            index_l = max(peakutils.indexes(hist_left, thres=0.5, min_dist=30))
        except:
            #plt.imshow(img, cmap='binary')
            #plt.show()
            index_l = np.argmax(hist_left)
            print(step_size, index_l)
        #print("Left Peaks: ", index_l, ", Values at peak points: ", hist_left[index_l])

        try:
            index_r = max(peakutils.indexes(hist_right, thres=0.5, min_dist=30))
        except:
            #plt.imshow(img, cmap='binary')
            #plt.show()
            #plt.plot(histogram)
            #plt.show()
            index_r = np.argmax(hist_right)
            print(step_size, index_r)
        #print("Right Peaks: ", index_r, ", Values at peak points: ", hist_right[index_r])

        index_r += index_r_offset
        index_l += index_l_offset
        #print("Left Peaks: ", index_l)
        #print("Right Peaks: ", index_r)

        histogram = np.sum(img[max(img.shape[0]//2-step_size,0):img.shape[0]-step_size,:], axis=0)
        hist_left = histogram[:,1][index_l-hist_step:index_l+hist_step] + histogram[:,2][index_l-hist_step:index_l+hist_step]
        hist_right = histogram[:,1][index_r-hist_step:index_r+hist_step] + histogram[:,2][index_r-hist_step:index_r+hist_step]
        index_l_offset = index_l-hist_step
        index_r_offset = index_r-hist_step

        left_lane_points = sliding_window(img, step_size, window_size, index_l, left_lane_points)
        right_lane_points = sliding_window(img, step_size, window_size, index_r, right_lane_points)

    #plt.scatter(*zip(*all_points))
    #plt.show()

    #compute fit
    left_lane.compute_fit(left_lane_points)
    right_lane.compute_fit(right_lane_points)

    return left_lane, right_lane 

def identify_lane_pixels(img, left_lane, right_lane):

    # use sliding window to identify rest of the lane
    window_size = 48
    left_lane_points=[]
    right_lane_points=[]
    all_points=[]

    for step_size in range(0, img.shape[0], window_size):

        y = img.shape[0]-step_size
        index_l = left_lane.get_x(y)
        index_r = right_lane.get_x(y)

        left_lane_points = sliding_window(img, step_size, window_size, index_l, left_lane_points)
        right_lane_points = sliding_window(img, step_size, window_size, index_r, right_lane_points)

    #plt.scatter(*zip(*all_points))
    #plt.show()

    #compute fit
    left_lane.compute_fit(left_lane_points)
    right_lane.compute_fit(right_lane_points)

    return left_lane, right_lane 

def draw_lanes(image, warped, Minv, left_lane, right_lane, vehicle_pos):

    # Create an empty image to draw the lines on
    color_warp = np.zeros_like(warped).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.allx, left_lane.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.allx, right_lane.ally])))])
    pts = np.hstack((pts_left, pts_right))

    #print(pts)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # print diagnostics onto the image
    text_color = (255,255,255)
    text_str = 'Radius of curvature(left): %dm'%(left_lane.radius_of_curvature)
    cv2.putText(result, text_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color)
    text_str = 'Radius of curvature(right): %dm'%(right_lane.radius_of_curvature)
    cv2.putText(result, text_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color)

    if (vehicle_pos) < 0:
        text_str = 'Vehicle is %5.2fm left of center'%(-vehicle_pos)
    elif (vehicle_pos) > 0:
        text_str = 'Vehicle is %5.2fm right of center'%(vehicle_pos)
    else:
        text_str = 'Vehicle is at center'
    cv2.putText(result, text_str, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color)

    return result

class LaneDetector(object):

    to_plot = False

    def __init__(self):
        self.left_lane = Lane()
        self.right_lane = Lane()
        self.M = M
        self.Minv = Minv

    def __plot(self, img):
        if self.to_plot:
            plt.imshow(img, cmap='binary')
            plt.show()

    ''' Full pipeline to take in a camera image and return the lane-detected
        image
    '''
    def pipeline(self, img):

        # detect edges in the image
        color_binary = detect_edges(img, s_thresh=(170, 255), sx_thresh=(20, 100))

        # apply perspective transform only on the first image
        if not len(M):
            self.M, self.Minv = get_perspective_transform(img)
            print("Perspective transform calculated")

        # get a bird's eye view of the image
        img_bird_eye = do_birds_eye(color_binary, self.M)
        self.__plot(img_bird_eye)

        # identify the lane lines in the bird's eye image
        if self.left_lane.current_fit[0]:
            self.left_lane, self.right_lane = identify_lane_pixels(img_bird_eye, self.left_lane, self.right_lane)
        else:
            print("First video frame")
            self.left_lane, self.right_lane = blind_identify_lane_pixels(img_bird_eye, self.left_lane, self.right_lane)

        # compute radius of curvature
        self.left_lane.get_radius_curvature()
        self.right_lane.get_radius_curvature()

        # compute vehicle position
        left_pos = self.left_lane.get_vehicle_position()
        right_pos = self.right_lane.get_vehicle_position()
        self.vehicle_pos = left_pos + right_pos

        # draw detected lanes back onto original image
        output_image = draw_lanes(img, img_bird_eye, self.Minv, self.left_lane, self.right_lane, self.vehicle_pos)
        return output_image

images = glob.glob('./test_images/*.jpg')
print (images)

for fname in images:
    # create objects for the left and right lanes
    lane_detector = LaneDetector()

    # Read in image
    img = cv2.imread(fname)
    output_image = lane_detector.pipeline(img)
    if not len(M):
        M = lane_detector.M
        Minv = lane_detector.Minv
    #plt.imshow(output_image)
    #plt.show()

# Time to try the algorithm on a video stream
lane_detector = LaneDetector()
video_output = 'video_output2.mp4'
clip1 = VideoFileClip("project_video.mp4")
#clip1 = VideoFileClip("challenge_video.mp4")
video_clip = clip1.fl_image(lane_detector.pipeline) #NOTE: this function expects color images!!
video_clip.write_videofile(video_output, audio=False, write_logfile=False, verbose=True, threads=4)
