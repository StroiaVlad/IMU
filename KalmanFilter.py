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
        self.x_initial = np.array([P0[0], V0[0], E0[0]]).reshape(self.n, 1)  # the state vector X from the previous iteration
        self.x = np.array([P0[0], V0[0], E0[0]]).reshape(self.n, 1)  # the state vector X as a 9x1 vector [x, y, z, vx, vy, vz, phi, theta, psi].T
        # output matrix
        self.C = np.array([[2 / (dt ** 2), 0, 0, 1 / dt, 0, 0, 0, 0, 0],
                           [0, 2 / (dt ** 2), 0, 0, 1 / dt, 0, 0, 0, 0],
                           [0, 0, 2 / (dt ** 2), 0, 0, 1 / dt, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1 / dt, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1 / dt, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1 / dt]])

        self.P = 500 * np.eye(self.n)  # covariance state matrix
        # discrete noise model is used, dt is small
        self.var_w = 10  # variance of the process noise w E[w*w.T]
        self.Q = (self.var_w ^ 2) * np.dot(self.B, self.B.T)  # process noise covariance matrix E[v*v.T]
        self.R = R  # measurement noise covariance matrix

    # prediction step
    def aprioriEstimation(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)  # state vector estimation
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # state covariance estimation

    # update step
    def aposterioriEstimation(self, y):
        S = np.linalg.inv(np.dot(np.dot(self.C, self.P), self.C.T) + self.R)
        self.K = np.dot(np.dot(self.P, self.C.T), S)  # computing the Kalman Gain
        self.x = self.x + np.dot(self.K, (y - np.dot(self.C, self.x)))  # state vector estimation
        self.P = np.dot(np.eye(self.n) - np.dot(self.K, self.C), self.P)  # covariance matrix estimation

    def estimation(self, KF, y, GPS_time, observations):
        """estimation function returns the state vector containing all the states predicted by the KF Kalman Filter, based on the
        y measurements matrix and u input command vector"""
        g = np.empty(shape=[3, 1])
        state = np.empty(shape=[KF.n, 1])  # initializing the state covariance vector as a column vector of KF.n rows
        # covariance = np.empty(shape=[KF.n, 1])  # initializing the covariance matrix as a
        for index in range(0, y.shape[0]):  # iterating from 0 till the number of data points in the measurements matrix
            u = np.array(y[index].reshape(y.shape[1], 1))  # aprioriEstimation function requires an u column vector of shape (states measured by sensors)x1, that's why reshape is used
            if index < y.shape[0] - 1:
                u[3:, :] = ECEF02ECEF(u[3:, :], GPS_time[index + 1] - GPS_time[index])  # converting the gyro data from ECEF0 to ECEF
            if index == y.shape[0] - 1:
                u[3:, :] = ECEF02ECEF(u[3:, :], GPS_time[index] - GPS_time[index - 1])  # converting the gyro data from ECEF0 to ECEF
            rot_ECEF2BODY = ECEF2body(u[0:3, :], KF.x[6][0], KF.x[7][0], KF.x[8][0])  # rotation matrix from ECEF to BODY
            rot_ECEF2BODY = np.array(list(rot_ECEF2BODY[:, :]), dtype=np.float64)  # making a numpy array out of the rotation matrix
            u[0:3, :] = np.dot(np.linalg.inv(rot_ECEF2BODY), u[0:3, :])  # converting the Specific force from BODY frame to ECEF frame
            gravity_vector = np.array([[0], [0], [9.809792]])   # gravity vector in ENU frame
            [phi_reference, lamda_reference, h_reference] = ECEF2LLA(gravity_vector)    #
            rot_ECEF2ENU = np.zeros((3, 3)) # defining the rotation matrix
            rot_ECEF2ENU[0, :] = [-np.sin(lamda_reference), np.cos(lamda_reference), 0]
            rot_ECEF2ENU[1, :] = [-np.sin(phi_reference) * np.cos(lamda_reference),
                                  -np.sin(phi_reference) * np.sin(lamda_reference), np.cos(phi_reference)]
            rot_ECEF2ENU[2, :] = [np.cos(phi_reference) * np.cos(lamda_reference),
                                  np.cos(phi_reference) * np.sin(lamda_reference), np.sin(phi_reference)]
            gravity_ECEF = np.dot(np.linalg.inv(rot_ECEF2ENU), gravity_vector)  # converting the gravity vector from ENU to ECEF
            u[0:3, :] = u[0:3, :] + gravity_ECEF  # removing the gravity component from the accelerometer data
            KF.aprioriEstimation(u)  # apply the apriori estimation to the KF
            KF.aposterioriEstimation(observations)  # apply the aposteriori estimation to the KF
            state = np.concatenate([state, KF.x], axis=1)  # concatenate a new column of predicted state vector (KF.x) to the state => state vector becomes a state matrix
            KF.x_initial = KF.x  # updating the KF.x_initial to the currently estimated KF.x
            g = np.concatenate([g, gravity_ECEF], axis=1)
            # covariance = np.concatenate([state, KF.P], axis=1)
        state = np.delete(state, 0, axis=1)
        g = np.delete(g, 0, axis=1)
        print(g)
        # covariance = np.delete(covariance, 0, axis=1)
        return state
