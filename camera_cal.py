'''
SDC P4 - Advanced Lane Lines Detection
Author: Soujanya Kedilaya
'''

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Universal parameters
nx = 9
ny = 6
nchannels = 3

''' Camera calibration from set of test images '''

# Array to store object and image points from all test images
objpoints = [] #3D points in real world space
imgpoints = [] #2D points in test images

plot_images = []

objp = np.zeros((nx*ny,nchannels), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

''' Function to plot images for data visualization '''
def plot_images_fxn(images):
    
    # Create figure with 5x4 sub-plots.
    fig, axes = plt.subplots(5, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    plt.title("Camera cal images")
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
            
        # Plot image.
        ax.imshow(images[i], cmap='binary')

        #xlabel = "Steering angle: {0}".format(angles[i])

        # Show the angles as the label on the x-axis.
        #ax.set_xlabel(xlabel)
        #ax.set_title(labels[i])
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.savefig('camera_calibrated.png', bbox_inches='tight')
    plt.close()

images = glob.glob('./camera_cal/calibration*.jpg')
print (images)
for fname in images:
    # Read in image
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        imgpoints.append(corners.astype('float32'))
        objpoints.append(objp.astype('float32'))

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plot_images.append(img)
    else:
        print ("Failed to find chessboard corners: ", fname)

# Plot cal images with chessboard corners
plot_images_fxn(plot_images)

# Calibrate the camera!
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print (ret)

# Save the data for easy access
pickle_file = 'camera_cal.pickle'
print('Saving cal data to pickle file...')
try:
    with open('camera_cal.pickle', 'wb') as pfile:
        pickle.dump(
            {
                'mtx': mtx,
                'dist': dist,
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
print('Data cached in pickle file.')

