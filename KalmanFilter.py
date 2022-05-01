import numpy as np
from conversion import *

class KalmanFilter:
    def __init__(self, P0, V0, E0, R, dt):
        # state matrix
        self.A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, dt, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, dt, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # control matrix
        self.B = np.array([[0 * 0.5 * dt * dt, 0, 0, 0, 0, 0],
                           [0, 0 * 0.5 * dt * dt, 0, 0, 0, 0],
                           [0, 0, 0 * 0.5 * dt * dt, 0, 0, 0],
                           [dt, 0, 0, 0, 0, 0],
                           [0, dt, 0, 0, 0, 0],
                           [0, 0, dt, 0, 0, 0],
                           [0, 0, 0, dt, 0, 0],
                           [0, 0, 0, 0, dt, 0],
                           [0, 0, 0, 0, 0, dt]])
        self.n = np.shape(self.A)[0]  # row dimension of the A matrix (9 states)
        self.x_initial = np.array([P0[0], V0[0], E0[0] * 180/np.pi]).reshape(self.n, 1)  # the state vector x from the previous iteration
        # E0[0] [heading, attitude, roll] converted into radians
        self.x = np.array([P0[0], V0[0], E0[0] * np.pi/180]).reshape(self.n, 1)  # the state vector X as a 9x1 vector [x, y, z, vx, vy, vz, psi, theta, phi].T in ECEF frame
        self.innovation = np.zeros([6, 1]) # innovations
        # output matrix
        self.C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0]])
        self.P = np.eye(self.n)  # covariance state matrix
        # discrete noise model is used, dt is small
        #self.var_w = 10  # variance of the process noise w = E[w*w.T]
        #self.Q = (self.var_w ^ 2) * np.dot(self.B, self.B.T)  # process noise covariance matrix E[v*v.T], if noiseless IMU => Q almost 0
        self.Q = np.eye(self.n)
        self.R = R  # measurement noise covariance matrix

    # prediction step
    def aprioriEstimation(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)  # state vector estimation
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # state covariance estimation

    # update step
    def aposterioriEstimation(self, y):
        S = np.linalg.inv(np.dot(np.dot(self.C, self.P), self.C.T) + self.R)
        self.K = np.dot(np.dot(self.P, self.C.T), S)  # computing the Kalman Gain
        self.innovation = y - np.dot(self.C, self.x)
        self.x = self.x + np.dot(self.K, self.innovation)  # state vector estimation
        #print(y - np.dot(self.C, self.x))
        self.P = np.dot(np.eye(self.n) - np.dot(self.K, self.C), self.P)  # covariance matrix estimation

    def estimation(self, KF, measurements, GPS_time_IMU, dt, withGNSS, observations, GPS_time_GNSS):
        """estimation function returns the state vector containing all the states predicted by the KF Kalman Filter, based on the
        y measurements matrix and u input command vector"""
        g = np.empty(shape=[3, 1])  # g is the gravity matrix in ECEF frame, after a measurements is introduced a new 3x1 column vector is added to this matrix
        state = np.empty(shape=[KF.n, 1])  # initializing the state vector as a column vector of KF.n rows
        covariance = np.empty(shape=[KF.n, 1])  # initializing the covariance matrix as a

        GNSS_index = 0
        for IMU_index in range(0, measurements.shape[0]):  # iterating from 0 till the number of data points in the measurements matrix
            # u are the IMU measurements, where the specific forces are in the Body frame and the angular rates are in ECEF0 frame
            u = np.array(measurements[IMU_index].reshape(measurements.shape[1], 1))  # aprioriEstimation function requires an u column vector of shape (states measured by sensors)x1, that's why reshape is used
            # subtract the Earth's angular rate from the measured angular rates and convert them from ECEF0 to ECEF
            # 7.292115e-05 rads/s is the angular speed of the Earth's Rotation in inertial space
            u[3:, :] = u[3:, :] - [[0], [0], [7.292115e-05]]  # converting the gyro data from ECEF0 to ECEF
            # in the state vector x, the order is heading, roll, yaw, but in the ECEF2body the order is heading, attitude, roll
            rot_ECEF2BODY = ECEF2body(KF.x[0:3], KF.x[6][0], KF.x[7][0], KF.x[8][0]) # rotation matrix from ECEF to Body, heading, attitude, roll
            rot_ECEF2BODY = np.array(list(rot_ECEF2BODY[:, :]), dtype=np.float64)  # making a numpy array out of the rotation matrix
            u[0:3, :] = np.dot(np.linalg.inv(rot_ECEF2BODY), u[0:3, :])  # converting the Specific force from BODY frame to ECEF frame by using the rot_ECEF2BODY

            gravity_vector = np.array([[0], [0], [9.81]])   # gravity vector given in the ENU frame
            [phi_reference, lamda_reference, h_reference] = ECEF2LLA(KF.x[0:3])    # phi and lamda geodetic coordinates needed for ECEF to ENU rotation matrix
            rot_ECEF2ENU = np.zeros((3, 3)) # defining the rotation matrix from ECEF to ENU since there is no function ENU2ECEF in conversion file
            rot_ECEF2ENU[0, :] = [-np.sin(lamda_reference), np.cos(lamda_reference), 0]
            rot_ECEF2ENU[1, :] = [-np.sin(phi_reference) * np.cos(lamda_reference),
                                  -np.sin(phi_reference) * np.sin(lamda_reference), np.cos(phi_reference)]
            rot_ECEF2ENU[2, :] = [np.cos(phi_reference) * np.cos(lamda_reference),
                                  np.cos(phi_reference) * np.sin(lamda_reference), np.sin(phi_reference)]
            gravity_ECEF = np.dot(np.linalg.inv(rot_ECEF2ENU), gravity_vector)  # converting the gravity vector from ENU to ECEF
            u[0:3, :] = u[0:3, :] - gravity_ECEF  # removing the gravity component from the accelerometer data (since both are in ECEF frame)

            # in the state vector x, the order is heading, roll, yaw, but in the ECEF2body the order is heading, attitude, roll
            # actually we want the heading, roll and yaw in the state vector x to be expressed in the body frame, since they were converted from ECEF0 to ECEF
            KF.B[6:9, 3:6] = np.dot(np.array([[0, 0, dt], [0, dt, 0], [dt, 0 , 0]]), ECEF2body(KF.x[0:3], KF.x[6][0], KF.x[7][0], KF.x[8][0]))     # changing the B matrix

            KF.aprioriEstimation(u)  # apply the apriori estimation to the KF after the specific force was converted to ECEF and gyro data have been converted to Body frame

            if withGNSS and GNSS_index < observations.shape[0] and GPS_time_IMU[IMU_index] == GPS_time_GNSS[GNSS_index]:
                y = np.array(observations[GNSS_index].reshape(observations.shape[1], 1))
                #print(GNSS_index)
                KF.R = np.array([[y[6][0], 0, 0, 0, 0, 0],
                                [0, y[7][0], 0, 0, 0, 0],
                                [0, 0, y[8][0], 0, 0, 0],
                                [0, 0, 0, y[9][0], 0, 0],
                                [0, 0, 0, 0, y[10][0], 0],
                                [0, 0, 0, 0, 0, y[11][0]]])
                KF.aposterioriEstimation(y[0:6, :])  # apply the aposteriori estimation to the KF
                GNSS_index = GNSS_index + 1

            state = np.concatenate([state, KF.x], axis=1)  # concatenate a new column of predicted state vector (KF.x) to the state => state vector becomes a state matrix
            KF.x_initial = KF.x  # updating the KF.x_initial to the currently estimated KF.x
            g = np.concatenate([g, gravity_ECEF], axis=1)   # gravity_ECEF for this measurement is concatenated to the g matrix
            covariance = np.concatenate([state, KF.P], axis=1)
        state = np.delete(state, 0, axis=1) # delete the first column due to the fact that np.empty function initializes the first column with random elements
        g = np.delete(g, 0, axis=1)         # delete the first column due to the fact that np.empty function initializes the first column with random elements
        covariance = np.delete(covariance, 0, axis=1)
        return state
