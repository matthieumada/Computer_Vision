import cv2
import numpy as np
from mouseclick import getMouseClick

a4width = 210
a4length = 297


def computeHomography(corners_a4_image):
    # ----------------- YOUR CODE HERE ----------------------
    # Define the paper plane by defining 4 corner points in the plane (Using the A4 paper size)
    # Compute the homography matrix using 
        # 4 corner points in the image plane as srcPoints
        # 4 corner points in the A4 plane as dstPoints
    # You should use cv2.findHomography(srcPoints, dstPoints)
    # --------------------------------------------------------
    srcPoints =  np.array(corners_a4_image,dtype = np.float32)
    dstPoints = np.array([[ 0,0], # top-left
                           [0,a4length],# bottom-left
                           [a4width,a4length], # bottom-right
                           [a4width,0]],dtype = np.float32) # top-right
    H = cv2.findHomography(srcPoints, dstPoints)
    return H[0][0:3,0:3]

def warpImage(img, H):
    # ----------------- YOUR CODE HERE ----------------------
    # Define the size of the "paper" or target plane
    # Warp the image using the homography matrix
    # -------------------------------------------------------
    output_size = (a4width,a4length) # (width, height) in mm
    img_warp = cv2.warpPerspective(img,H,output_size)
    return img_warp

def computeWidthHeight(corners_pcb_original, H):
    # ----------------- YOUR CODE HERE ----------------------
    # Transform each point using the homography matrix
    corners = np.array(corners_pcb_original,dtype = np.float32)
    corners = np.array([corners]) # shape (1, 4, 2) for cv2.perspectiveTransform
    warped_corners = cv2.perspectiveTransform(corners, H)[0]  # shape (4, 2)
    # Compute the width and height of the PCB in the warped image
    width = np.linalg.norm(warped_corners[0]-warped_corners[1])
    height = np.linalg.norm(warped_corners[1]-warped_corners[2])
    # -------------------------------------------------------


    return width, height

def main():
    path = "/home/delinm/Documents/Robotics_Computer_Vision/Computer_Vision/LAB_4/"
    # Read the image
    img = cv2.imread(path + "PCB.jpg")
    
    corners_a4_image = []
    # ----------------- YOUR CODE HERE ----------------------
    # Get the 4 corners of the A4 paper in the image by clicking on them
    # You can use the getMouseClick function to get the corners: 
    left_up = getMouseClick("Click on A4 left up corners", img)
    corners_a4_image.append(left_up)
    left_down = getMouseClick("Click on A4 left down corners", img)
    corners_a4_image.append(left_down)
    right_down = getMouseClick("Click on A4 right down corner", img)
    corners_a4_image.append(right_down)
    right_up = getMouseClick("Click on A4 right up corners", img)
    corners_a4_image.append(right_up)
    # It is recommended to hardcode the corners after getting them once
    # -------------------------------------------------------
    print("Clicked A4 corners in the image: ", corners_a4_image)
    
    # Visualize corners in the original image
    for corner in corners_a4_image:
        cv2.circle(img, corner, 50, (255, 255, 255), 10)
    
    cv2.destroyAllWindows()


    corners_pcb_original = []
    # ----------------- YOUR CODE HERE ----------------------
    # Get the corners of the PCB in the image by clicking on them
    # You can use the getMouseClick function to get the corners
    # It is recommended to hardcode the corners after getting them once
    left_up = getMouseClick("Click on PCB left up corners", img)
    corners_pcb_original.append(left_up)
    left_down = getMouseClick("Click on PCB left down corners", img)
    corners_pcb_original.append(left_down)
    right_down = getMouseClick("Click on PCB right down corner", img)
    corners_pcb_original.append(right_down)
    right_up = getMouseClick("Click on PCB right up corners", img)
    corners_pcb_original.append(right_up)
    # -------------------------------------------------------
    print("Clicked PCB corners in the image: ", corners_pcb_original)

    # Visualize PCB corners
    for i, corner in enumerate(corners_pcb_original):
        cv2.circle(img, (corner[0], corner[1]), 50, ((i % 2) * 255, 0, (i // 2) * 255), 10)

    # Scale down image for visualization
    img_show = cv2.pyrDown(cv2.pyrDown(img))
    cv2.destroyAllWindows()
    # Show the original image
    cv2.imshow("Original", img_show)
    cv2.waitKey(0)

    # Compute the homography matrix
    H = computeHomography(corners_a4_image)
    print("H:\n", H)

    # Warp the image using the homography
    img_warp = warpImage(img, H)

    # Save the warped image
    cv2.imwrite(path + "PCB_warp2.jpg", img_warp)

    # Visualize the warped image
    img_warp_show = cv2.pyrDown(cv2.pyrDown(img_warp))
    cv2.imshow("Warp", img_warp_show)
    cv2.waitKey(0)

    # Compute the width and height after warping
    width, height = computeWidthHeight(corners_pcb_original, H)

    print(f"Width (in mm): {width}")
    print(f"Height (in mm): {height}")

    # Show final images for reference
    cv2.imshow("Original", img_show)
    cv2.imshow("Warp", img_warp_show)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
