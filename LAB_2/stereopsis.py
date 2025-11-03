import cv2
import numpy as np
import sys
from methods import loadImagesAndCalibration, getMouseClick

def constructProjectionMat(cam):
    KA = cam.intrinsic
    H = cam.transformation

    zeros = np.zeros((3, 1), dtype=np.float64)
    KA = np.hstack((KA, zeros))
    return np.dot(KA, H)

def splitPp(proj):
    Pp = [proj[:3, :3], proj[:3, 3:4]]
    return Pp

def computeOpticalCenter(Pp):
    one = np.ones((1, 1), dtype=np.float64)
    C = -np.dot(np.linalg.inv(Pp[0]), Pp[1])
    C = np.vstack((C, one))
    return C

def computeEpipole(proj, C):
    e = np.dot(proj, C)
    return e

def computeFundamentalMat(e, proj_r, proj_l):
    erx = np.array([
        [0,         -e[2, 0],   e[1, 0]],
        [e[2, 0],   0,          -e[0, 0]],
        [-e[1, 0],  e[0, 0],    0]
    ], dtype=np.float64)

    F_lr = np.dot(erx, np.dot(proj_r, np.linalg.pinv(proj_l)))
    return F_lr

def computeEpipolarLine(F, m):
    e_line = np.dot(F, m)
    return e_line

def drawEpipolarLine(img, line, width):
    x, y, z = line[0, 0], line[1, 0], line[2, 0]
    p1 = (0, int(-z / y))
    p2 = (int(width), int(-width * x / y - z / y))

    cv2.line(img, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

def computePluckerLine(M1, M2):
    plucker_0 = np.cross(M1.flatten(), M2.flatten()) / np.linalg.norm(M2)
    plucker_1 = M2 / np.linalg.norm(M2)
    plucker = [plucker_1, plucker_0]
    return plucker

def project2Inf(P, m):
    M_inf = np.dot(np.linalg.pinv(P), m)
    return M_inf

def computePluckerIntersect(plucker_1, plucker_2):
    # mu1, mu2, v1, v2 = plucker_1[0], plucker_2[0], plucker_1[1], plucker_2[1]
    mu1 = np.asarray(plucker_1[0]).flatten()
    mu2 = np.asarray(plucker_2[0]).flatten()
    v1 = np.asarray(plucker_1[1]).flatten()
    v2 = np.asarray(plucker_2[1]).flatten()

    v1_v2xmu2 = np.dot(v1, np.cross(v2, mu2))
    v1v2_v1_v2xmu1 = np.dot(np.dot(v1, v2), np.dot(v1, np.cross(v2, mu1)))
    pow_v1xv2 = np.linalg.norm(np.cross(v1, v2)) ** 2
    M1 = (v1_v2xmu2 - v1v2_v1_v2xmu1) / pow_v1xv2 * v1 + np.cross(v1, mu1)

    v2_v1xmu1 = np.dot(v2, np.cross(v1, mu1))
    v2v1_v2_v1xmu2 = np.dot(np.dot(v2, v1), np.dot(v2, np.cross(v1, mu2)))
    pow_v2xv1 = np.linalg.norm(np.cross(v2, v1)) ** 2
    M2 = (v2_v1xmu1 - v2v1_v2_v1xmu2) / pow_v2xv1 * v2 + np.cross(v2, mu2)

    return M1 + (M2 - M1) / 2

def main():
    # Load images and calibration
    img_l, img_r, stereoPair = loadImagesAndCalibration(sys.argv)
    if img_l is None or img_r is None or stereoPair is None:
        print("Input error")
        return

    # Construct projection matrices
    proj_l = constructProjectionMat(stereoPair.cam1)
    proj_r = constructProjectionMat(stereoPair.cam2)
    print("Camera 1 (left) projection matrix", proj_l)
    print("Camera 2 (right) projection matrix", proj_r)

    # Compute optical centers
    Pp_l = splitPp(proj_l)
    Pp_r = splitPp(proj_r)
    C_l = computeOpticalCenter(Pp_l)
    C_r = computeOpticalCenter(Pp_r)
    print("Optical center Left:", C_l)
    print("Optical center Right:", C_r)

    # Compute epipoles
    e_l = computeEpipole(proj_l, C_r)
    e_r = computeEpipole(proj_r, C_l)
    print("Left epipole:", e_l)
    print("Right epipole:", e_r)

    # Compute fundamental matrix
    F_lr = computeFundamentalMat(e_r, proj_r, proj_l)
    print("Fundamental matrix left to right:", F_lr)

    # Get mouse click on a point in the left image
    m_l = getMouseClick("LeftImage", img_l)

    # Compute epipolar line in the right image, and draw it
    e_line_r = computeEpipolarLine(F_lr, m_l)
    drawEpipolarLine(img_r, e_line_r, stereoPair.cam2.image_width)

    # Get mouse click on the same point in the right image
    m_r = getMouseClick("RightImage", img_r)

    # Project the points to infinity
    M_inf_l = project2Inf(Pp_l[0], m_l)
    M_inf_r = project2Inf(Pp_r[0], m_r)
    print("M_inf left:", M_inf_l)
    print("M_inf right:", M_inf_r)


    # Compute the plucker lines and their intersection
    plucker_l = computePluckerLine(C_l[:3, :1], M_inf_l)
    plucker_r = computePluckerLine(C_r[:3, :1], M_inf_r)
    print("Plucker line left:", plucker_l)
    print("Plucker line right:", plucker_r)
    intersect = computePluckerIntersect(plucker_l, plucker_r)

    print("Plucker intersection / Triangulated point:", intersect)

if __name__ == "__main__":
    main()
# to launch the code copy this:
#python3 stereopsis.py \./artificialImages/non-rectified/calibration.txt \./artificialImages/non-rectified/left.png \./artificialImages/non-rectified/right.png
