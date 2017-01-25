# Standard modules
import pickle
import sys
import cv2
import glob
import numpy as np

# Plotting modules
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Import Lane class
from Lane import Lane

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("camera_cal.pickle", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
M = []
Minv = []

# Debug globals
g_debug_plot = False
g_debug_log = False

# Notes -- Ignore, just for reference
# img.shape[0] = 720, img.shape[1] = 1280

''' Function: Take in 4 src points and modify them to obtain a birds-eye
    transformation function 
'''
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
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

    # the new src points can now be projected to obtain a proper bird's eye
    # view
    src = np.array([[tl[0], tl[1]], [tr[0], tr[1]], [br[0], br[1]], [bl[0], bl[1]]], \
            dtype = "float32")

    # now that we have the new src points, construct the set of destination
    # points to obtain a "birds eye view", (i.e. top-down view) of the image,
    # again specifying points in the top-left, top-right, bottom-right, and
    # bottom-left order
    dst = np.array([
            [img.shape[1]/4-20, bl[1]/6], 
            [img.shape[1]*3/4-10, br[1]/6], 
            [img.shape[1]*3/4-10, br[1]], 
            [img.shape[1]/4-20, bl[1]]], dtype = "float32")


    # log src and dst points
    print ('src: ', src, '\ndst: ', dst)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    Minv = cv2.getPerspectiveTransform(dst, pts)

    # return the warped image
    return M, Minv

''' Function: Get a perspective transform given 4 src points 
'''
def get_perspective_transform(img):
    if g_debug_plot:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(img, cmap='binary')
        plt.grid(True)
        plt.grid(b=True, which='major', color='b', linestyle='-')
        plt.grid(b=True, which='minor', color='r', linestyle='--')
        plt.minorticks_on()
        plt.show()

    # assuming this is on an image of size 1280x720 (tl, tr, br, bl)
    pts = np.array([(625, img.shape[0]-300),
                    (662, img.shape[0]-300),
                    (1125,img.shape[0]),
                    (200, img.shape[0])], dtype = "float32") #full lane
     
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    M, Minv = four_point_transform(img, pts)
    return M, Minv

''' Function: Apply the perspective transform on to the original image 
'''
def do_birds_eye(img, M):
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    if g_debug_plot:
        plt.imshow(warped)
        plt.savefig('warped.png')
        plt.close()
    return warped
   
''' Function: Detect gradient features such as lines and edges in an image 
'''
def detect_edges(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    undist = cv2.undistort(img, mtx, dist, None, mtx)    
    img = np.copy(undist)
    
    # Source for the below code: Udacity lecture notes

    # Convert to HLS color space and separate the channels
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
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    img_size = color_binary.shape[::-1]
    if g_debug_plot:
        plt.imshow(color_binary)
        plt.savefig('color_binary.png')
        plt.close()
    return color_binary

''' Create a sliding window and find all lane points within it
'''
def sliding_window(img, step_size, window_size, index, lane_points):
    
    # define a sliding window of 'window_size' 
    window = img[img.shape[0]-(step_size+window_size):img.shape[0]-step_size, \
                 index - window_size/2:index + window_size/2]

    # extract the pixel values within the window
    l_channel = window[:,:,1]
    s_channel = window[:,:,2]

    # collect the positions of the non-zero pixels within the window, since
    # they correspond to features in the image. Assume they belong to the lane
    # lines
    for i in range(window_size):
        for j in range(window_size):
            if window.shape[0] != window_size or window.shape[1] != window_size:
                print("Hey, we have a prob!", step_size, index)

            if(l_channel[i, j] != 0 or s_channel[i, j] != 0):
                x_coord = index - window_size/2 + j
                y_coord = img.shape[0]-(step_size+window_size) + i
                lane_points.append((x_coord,y_coord)) 
                #all_points.append((x_coord,y_coord)) 
    
    # return the lane positions
    return lane_points

''' Function: Find the peaks in the histogram of a full image to get a starting
    point for lane detection 
'''
def blind_search(img):
    # histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    # splitting the above histogram into left and right sections
    hist_len_half = int(len(histogram)/2)
    hist_left = histogram[:,1][0:hist_len_half] + histogram[:,2][0:hist_len_half]
    hist_right = histogram[:,1][hist_len_half:len(histogram)] + histogram[:,2][hist_len_half:len(histogram)]

    # Identify peaks in the histogram and get first estimate of lane start
    index_l = max(peakutils.indexes(hist_left, thres=0.5, min_dist=30))
    index_r = max(peakutils.indexes(hist_right, thres=0.5, min_dist=30))
    index_r += hist_len_half

    if g_debug_log:
        x = np.linspace(0, hist_len_half-1, hist_len_half)
        print("Left Peaks: ", index_l, ", Values at peak points: ", x[index_l], hist_left[index_l])
        print("Right Peaks: ", index_r, ", Values at peak points: ", x[index_r], hist_right[index_r])

    if g_debug_plot:
        x = np.linspace(0, hist_len_half-1, hist_len_half)
        plt.figure(figsize=(10,6))
        pplot(x, histogram[:,1], indexes)
        plt.title('First estimate')
        plt.show()

    return index_l, index_r, hist_len_half

''' Function: Get peak from histogram 
'''
def get_peak(hist, img, step_size):
    try:
        index = max(peakutils.indexes(hist, thres=0.5, min_dist=30))
    except:
        # unable to find peak, just use simple max of histogram
        index = np.argmax(hist)

        if g_debug_plot:
            plt.imshow(img, cmap='binary')
            plt.show()
        if g_debug_log:
            print(step_size, index)

    if g_debug_log:
        print("Peaks: ", index, ", Values at peak points: ", hist[index])

    return index

''' Function: Given an image, find all lane pixels with no previous lane
    information
'''
def blind_identify_lane_pixels(img, left_lane, right_lane):

    print ("Performing a blind search")
    index_l, index_r, hist_len_half = blind_search(img)

    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    hist_len_half = int(len(histogram)/2)

    # use sliding window to identify rest of the lane
    hist_step = 60 #pixels
    window_size = 48
    left_lane_points=[]
    right_lane_points=[]
    left_lane.detected = False
    right_lane.detected = False  

    for step_size in range(0, img.shape[0], window_size):

        # use the previous discovered peaks and corresponding x-coords as starting
        # point to slide up the image
        histogram = np.sum(img[max(img.shape[0]//2-step_size,0):img.shape[0]-step_size,:], axis=0)
        hist_left = histogram[:,1][index_l-hist_step:index_l+hist_step] + histogram[:,2][index_l-hist_step:index_l+hist_step]
        hist_right = histogram[:,1][index_r-hist_step:index_r+hist_step] + histogram[:,2][index_r-hist_step:index_r+hist_step]
        index_l_offset = index_l-hist_step
        index_r_offset = index_r-hist_step

        # get peaks
        index_l = (get_peak(hist_left, img, step_size)) + index_l_offset
        index_r = (get_peak(hist_right, img, step_size)) + index_r_offset

        # get lane points from within window
        left_lane_points = sliding_window(img, step_size, window_size, index_l, left_lane_points)
        right_lane_points = sliding_window(img, step_size, window_size, index_r, right_lane_points)

    # scatter plot the identified lane points
    if g_debug_plot:
        all_points = left_lane_points + right_lane_points
        plt.scatter(*zip(*all_points))
        plt.gca().invert_yaxis() # to visualize as we do the images        
        plt.show()

    # compute fit
    left_lane.compute_fit(left_lane_points, 'red')
    right_lane.compute_fit(right_lane_points, 'blue')

    return left_lane, right_lane 

''' Function: Given an image and previous lane information, use that as a
    starting point to find the updated lane info
'''
def identify_lane_pixels(img, left_lane, right_lane):

    # use sliding window to identify rest of the lane
    window_size = 48
    left_lane_points=[]
    right_lane_points=[]
    left_lane.detected = False
    right_lane.detected = False  

    for step_size in range(0, img.shape[0], window_size):

        y = img.shape[0]-step_size
        index_l = left_lane.get_x(y)
        index_r = right_lane.get_x(y)

        left_lane_points = sliding_window(img, step_size, window_size, index_l, left_lane_points)
        right_lane_points = sliding_window(img, step_size, window_size, index_r, right_lane_points)

    # scatter plot the identified lane points
    if g_debug_plot:
        all_points = left_lane_points + right_lane_points
        plt.scatter(*zip(*all_points))
        plt.gca().invert_yaxis() # to visualize as we do the images        
        plt.show()

    #compute fit
    left_lane.compute_fit(left_lane_points)
    right_lane.compute_fit(right_lane_points)

    return left_lane, right_lane 

''' Function: Draw the detected lanes back onto the original image
'''
def draw_lanes(image, warped, Minv, left_lane, right_lane):

    # Create an empty image to draw the lines on
    color_warp = np.zeros_like(warped).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.bestx, left_lane.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx, right_lane.ally])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, new_warp, 0.3, 0)
    return result

''' Function: Add diagnostic info to the image
'''
def add_diag_text(img, lane_roc, vehicle_pos):

    # Print diagnostics onto the image
    text_color = (255,255,255)
    text_str = 'Radius of curvature: %dm'%(lane_roc)
    cv2.putText(img, text_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color)

    if (vehicle_pos) < 0:
        text_str = 'Vehicle is %5.2fm left of center'%(-vehicle_pos)
    elif (vehicle_pos) > 0:
        text_str = 'Vehicle is %5.2fm right of center'%(vehicle_pos)
    else:
        text_str = 'Vehicle is at center'
    cv2.putText(img, text_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color)

    return img

''' Class: Defines an instance of a lane detector. Holds info on the left and
    right lanes and a pipeline to identify such lanes
'''
class LaneDetector(object):

    to_plot = g_debug_plot

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

        img_e = do_birds_eye(img, self.M)

        # identify the lane lines in the bird's eye image
        if self.left_lane.detected:
            self.left_lane, self.right_lane = identify_lane_pixels(img_bird_eye, self.left_lane, self.right_lane)
        else:
            self.left_lane, self.right_lane = blind_identify_lane_pixels(img_bird_eye, self.left_lane, self.right_lane)

        # smooth the detected lines over the last n frames
        self.left_lane.smooth_fit()
        self.right_lane.smooth_fit()

        #print(self.left_lane.best_fit, self.left_lane.current_fit)
        #print(self.right_lane.best_fit, self.right_lane.current_fit)

        # compute radius of curvature
        self.left_lane.get_radius_curvature()
        self.right_lane.get_radius_curvature()
        roc = (self.left_lane.radius_of_curvature + self.right_lane.radius_of_curvature)/2

        # compute vehicle position
        left_pos = self.left_lane.get_vehicle_position()
        right_pos = self.right_lane.get_vehicle_position()
        self.vehicle_pos = left_pos + right_pos

        # draw detected lanes back onto original image
        output_image = draw_lanes(img, img_bird_eye, self.Minv, self.left_lane, self.right_lane)

        # add the diagnostic info to the final image
        output_image = add_diag_text(output_image, roc, self.vehicle_pos)

        return output_image

''' ***************************** '''
''' Execute pipeline on test data '''
''' ***************************** '''

# Run the pipeline on the test images
images = glob.glob('./test_images/*.jpg')
print (images)

for fname in images:
    # create objects for the left and right lanes
    lane_detector = LaneDetector()

    # read in image
    img = cv2.imread(fname)

    # execute pipeline
    output_image = lane_detector.pipeline(img)
    if not len(M):
        M = lane_detector.M
        Minv = lane_detector.Minv
    if g_debug_plot:
        plt.imshow(output_image)
        plt.show()

    # save output image
    cv2.imwrite('output_images/output_{}'.format(fname.split("/")[2]), output_image)

# Time to try the algorithm on a video stream
lane_detector = LaneDetector()
video_output = 'video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
#clip1 = VideoFileClip("challenge_video.mp4")
video_clip = clip1.fl_image(lane_detector.pipeline) 
video_clip.write_videofile(video_output, audio=False, threads=4)
