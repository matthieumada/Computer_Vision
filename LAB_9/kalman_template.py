import numpy as np
import matplotlib.pyplot as plt
import cv2
from draw_utils import draw_crosses, draw_trajec, draw_line, draw_lines, draw_legend

# Constants
X_0     = 0     # Initial x position
Y_0     = 0     # Initial y position
V_0     = 100   # Initial velocity
ALPHA   = 1.2   # Launch angle
G       = -9.82 # Gravity
T_0     = 0     # Initial time
DT      = 0.25  # Time step

X_NOISE = 50
Y_NOISE = 0
PROCESS_NOISE = 0.1
MEASUREMENT_NOISE = 10# 

# Measurement big -> noise the filter doesn't trust the measure 
# Measurement small -> the filter trusts the measure
def calc_x(t : float, noise : float = 0) -> float:
    '''Calculate x position at time t with noise'''
    # ------------------------------------------------------------------------------------
    # ------------------------------ INSERT CODE HERE ------------------------------------
    pos_x = V_0*np.cos(ALPHA)*t + X_0 + noise 
    # ------------------------------------------------------------------------------------
    return pos_x

def calc_y(t : float, noise : float = 0) -> float:
    '''Calculate y position at time t with noise'''
    # ------------------------------------------------------------------------------------
    # ------------------------------ INSERT CODE HERE ------------------------------------
    pos_y = 0.5*G* t**2 + V_0*np.sin(ALPHA)*t + Y_0 + noise
    # ------------------------------------------------------------------------------------
    return pos_y

def position_kf():
    '''Create a Kalman filter with transition model based on position only'''
    kf = cv2.KalmanFilter(2, 2)
    # Matrix A
    kf.transitionMatrix = np.array([
            [1, 0], 
            [0, 1]], np.float32)
    # Initial position 
    kf.statePost = np.array([
            [calc_x(T_0)], 
            [calc_y(T_0)]], np.float32)
    kf.measurementMatrix = np.eye(2, 2, dtype=np.float32)
    kf.processNoiseCov = np.eye(2, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    return kf

def velocity_kf():
    ''''Create a Kalman filter with transition model based on position and velocity'''
    # ------------------------------------------------------------------------------------
    # ------------------------------ INSERT CODE HERE ------------------------------------
    kf_v = cv2.KalmanFilter(4, 2)
    # explanation: 4 size of the state vector, 2 size of output vector 
    # Kalman filter discrete :   X_k = A*X_k-1 + B*U_k + W_k 
    # Z_k = H*X_k + V_k

    # A matrix
    kf_v.transitionMatrix = np.array([
            [1, 0, DT, 0], 
            [0, 1, 0, DT],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)
    # Initial Posiiton 
    kf_v.statePost = np.array([
            [calc_x(T_0)], 
            [calc_y(T_0)],
            [V_0*np.cos(ALPHA)],
            [V_0*np.sin(ALPHA)]], np.float32)
    # H matrix of ouput
    kf_v.measurementMatrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0]], dtype=np.float32)
    kf_v.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kf_v.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    # ------------------------------------------------------------------------------------
    return kf_v

def acceleration_kf():
    '''Create a Kalman filter with transition model based on position, velocity and acceleration'''
    # ------------------------------------------------------------------------------------
    # ------------------------------ INSERT CODE HERE ------------------------------------
     # explanation: 6 size of the state vector, 2 size of output vector 
    # Kalman filter discrete :   X_k = A*X_k-1 + B*U_k + W_k 
    # Z_k = H*X_k + V_k

    kf_a = cv2.KalmanFilter(6,2)
    #  A Matrix
    kf_a.transitionMatrix = np.array([
            [1, 0, DT,  0, 0.5*DT**2,     0], 
            [0, 1,  0, DT,     0, 0.5 * DT**2],
            [0, 0,  1,  0,    DT,     0],
            [0, 0,  0,  1,     0,    DT],
            [0, 0,  0,  0,     1,     0],
            [0, 0,  0,  0,     0,     1]], np.float32)
    
    # Initial Position
    kf_a.statePost = np.array([
            [calc_x(T_0)], 
            [calc_y(T_0)],
            [V_0*np.cos(ALPHA)],
            [V_0*np.sin(ALPHA)],
            [0],
            [G]], np.float32)
    # H matrix of output 
    kf_a.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0]], dtype=np.float32)
    kf_a.processNoiseCov = np.eye(6, dtype=np.float32) * PROCESS_NOISE
    kf_a.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    # ------------------------------------------------------------------------------------
    return kf_a
   
def predict(kf : cv2.KalmanFilter):
    '''Call the prediction step of the Kalman filter'''
    # ------------------------------------------------------------------------------------
    # ------------------------------ INSERT CODE HERE ------------------------------------
    # ------------------------------------------------------------------------------------
    # Hint: Tuple, like so: (int(prediction[0, 0]), int(prediction[1, 0]))
    pred = kf.predict()
    x = int(pred[0,0])
    y = int(pred[1,0])
    return (x,y)

def correct(kf : cv2.KalmanFilter, measurement):
    # ------------------------------------------------------------------------------------
    # ------------------------------ INSERT CODE HERE ------------------------------------
    # ------------------------------------------------------------------------------------
    kf.correct(measurement)

def skip_correct(kf : cv2.KalmanFilter):
    '''Skip the correction step of the Kalman filter
    This is used to simulate a period where no measurements are available'''
    # ------------------------------------------------------------------------------------
    # ------------------------------ INSERT CODE HERE ------------------------------------
    kf.statePost = kf.statePre # spread the prediction
    kf.errorCovPost = kf.errorCovPre

    # ------------------------------------------------------------------------------------

def l2_error(pt1, pt2):
    '''Calculate the L2 error between two points'''
    # ------------------------------------------------------------------------------------
    # ------------------------------ INSERT CODE HERE ------------------------------------
    # ------------------------------------------------------------------------------------
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    norm = np.sqrt(dx*dx+ dy*dy)
    return float(norm)

def main():
    kf1 = position_kf()
    kf2 = velocity_kf()
    kf3 = acceleration_kf()

    cv2.namedWindow("Ball trajectory and Kalman filter")
    cv2.moveWindow("Ball trajectory and Kalman filter", 0, 450)

    # Lists to store history of points
    gt_points = []
    meas_points = []
    kf1_points = []
    kf2_points = []
    kf3_points = []
    error_kf1 = []
    error_kf2 = []
    error_kf3 = []
    time = []

    t = T_0
    while t < 30:
        gt = (int(calc_x(t)), int(calc_y(t)))
        meas = np.array([[calc_x(t, X_NOISE)], [calc_y(t, Y_NOISE)]], np.float32)
        meas_pt = (int(meas[0]), int(meas[1]))

        # Store the ground truth and measurement points
        gt_points.append(gt)
        meas_points.append(meas_pt)

        # Prediction and correction steps
        kf1_pred = predict(kf1)
        kf2_pred = predict(kf2)
        kf3_pred = predict(kf3)

        # Store the Kalman filter points for drawing
        kf1_points.append(kf1_pred)
        kf2_points.append(kf2_pred)
        kf3_points.append(kf3_pred)
        #-----------------

        # Compute the error from each version

        error_kf1.append(l2_error(kf1_pred, gt_points[0]))
        error_kf2.append(l2_error(kf2_pred, gt_points[0]))
        error_kf3.append(l2_error(kf3_pred, gt_points[0]))

        if t < 12 or t > 15:
            correct(kf1, meas)
            correct(kf2, meas)
            correct(kf3, meas)
        else:
            skip_correct(kf1)
            skip_correct(kf2)
            skip_correct(kf3)

        # Draw all points
        trajec = draw_trajec()
        trajec = draw_lines(trajec, gt_points, meas_points, kf1_points, kf2_points, kf3_points)

        # Draw crosses for the current points
        draw_crosses(trajec, meas_pt, gt, kf1_pred, kf2_pred, kf3_pred)

        # Flip and show the trajectory
        trajec = cv2.flip(trajec, 0)

        # Add a legend to the window
        trajec = draw_legend(trajec)

        # Show the trajectory
        cv2.imshow("Ball trajectory and Kalman filter", trajec)

        # ------------------------------------------------------------------------------------
        # ------------------------------ INSERT CODE HERE ------------------------------------
        '''Bonus: Plot the error between the ground truth and the Kalman filter predictions'''


        # ------------------------------------------------------------------------------------

        if cv2.waitKey(250) > 0:
            break
        time.append(t)
        t += DT
    
    # baby version 
    plt.figure()
    plt.title("Error between the ground truth and each Kalman filter predictions")
    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.plot(time, error_kf1, label="Kf_1")
    plt.plot(time, error_kf2, label="Kf_2")
    plt.plot(time, error_kf3, label="Kf_3")
    plt.grid(True)
    plt.legend()
    plt.savefig("error_kalman.png",format='png')
    plt.show()

    cv2.waitKey(0)



if __name__ == "__main__":
    main()
