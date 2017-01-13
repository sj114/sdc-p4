import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("camera_cal.pickle", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

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
    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    # return the warped image
    return M

def pipeline(img, do_perspective, M, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
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
    print ("Full image size: ", img_size)

    warped = color_binary[300:750, 100:1100] # [y1:y2, x1:x2] 

    if do_perspective:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped_gray = gray[300:750, 100:1100] # [y1:y2, x1:x2] 
        img_size_cropped = cropped_gray.shape[::-1]
        print ("img_size_cropped: ", img_size_cropped)
        plt.imshow(color_binary, cmap='binary')
        plt.grid(True)
        plt.grid(b=True, which='major', color='b', linestyle='-')
        plt.grid(b=True, which='minor', color='r', linestyle='--')
        plt.minorticks_on()
        plt.show()

        # this works on the cropped image of [300:750, 100:1100]
        #pts = np.array([(320,50),(450,50),(690,200),(100,200)], dtype = "float32")
         
        # this is on whole image of 960x540 (tl, tr, br, bl)
        #pts = np.array([(430,340),(530,340),(750,480),(250,480)], dtype = "float32")
         
        # this is on whole image of 1280x720 (tl, tr, br, bl)
        pts = np.array([(600,440),(700,440),(960,620),(350,620)], dtype = "float32")
         
        # apply the four point tranform to obtain a "birds eye view" of
        # the image
        M = four_point_transform(color_binary, pts)

    warped = cv2.warpPerspective(color_binary, M, (650,330))
    #src = np.float32([[680,300], [1100,750], [100,750], [600,300]])
    #dst = np.float32([[1100,300], [1100,750], [100,750], [100,300]])
    #src = np.float32([[750,240],[70,240],[370,0],[390,0]])
    #dst = np.float32([[600,240], [220,240], [220,0], [600,0]])
    #M = cv2.getPerspectiveTransform(src, dst)
    #warped = cv2.warpPerspective(cropped_gray, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M
    
images = glob.glob('./test_images/*.jpg')
print (images)
do_perspective = True
M = []
for fname in images:
    # Read in image
    img = cv2.imread(fname)
    result, M = pipeline(img, do_perspective, M)
    do_perspective = False 
    plt.imshow(result, cmap='binary')
    plt.show()
