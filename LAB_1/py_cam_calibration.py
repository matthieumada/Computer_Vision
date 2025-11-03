

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./Images/*.jpg') # if image name retunr ony one || for all image use this: ./Images/*.jpg'

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)

        if fname == './Images/left12.jpg':
            cv.imwrite('basic_image.png', img)

        cv.imshow('image', img)
        cv.waitKey(0)

#cv.destroyAllWindows()
### get camera calibration matrix
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Matrix:",mtx, "\tdistortion:",dist,"\nrotation vector ",rvecs,"\ntranslation vector",tvecs,"\nerror of reprojection",ret)

# Refine camera matrix 
img = cv.imread(images[3])
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print("New camera matrix:",newcameramtx,"\nROI:",roi)

###  -- Undistortion images ---
# -  With cv.undistort() - 
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('undistorted_image.png', dst)
cv.imshow('Undistortion method 1 ', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# - With remapping - 
# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst_2 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x2, y2, w2, h2 = roi
dst_2 = dst_2[y2:y2+h2, x2:x2+w2]
cv.imshow('Undistortion method 2 ', dst_2)
cv.waitKey(0)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
