import numpy as np
import pandas as pd


class KalmanFilter:
    def __init__(self, P0, V0, E0, R, dt):
        self.A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, dt, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, dt, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.B = np.array([[0.5*dt*dt, 0, 0, 0, 0, 0],
                          [0, 0.5*dt*dt, 0, 0, 0, 0],
                          [0, 0, 0.5*dt*dt, 0, 0, 0],
                          [dt, 0, 0, 0, 0, 0],
                          [0, dt, 0, 0, 0, 0],
                          [0, 0, dt, 0, 0, 0],
                          [0, 0, 0, dt, 0, 0],
                          [0, 0, 0, 0, dt, 0],
                          [0, 0, 0, 0, 0, dt]])
        self.n = np.shape(self.A)[0]                                # row dimension of the A matrix (9 states)
        self.x_initial = np.array([P0[0], V0[0], E0[0]]).reshape(self.n, 1)
        self.x = np.array([P0[0], V0[0], E0[0]]).reshape(self.n, 1) # state vector X as a 9x1 vector

        self.C = np.array([[0, 0, 0, (self.x[3][0] - self.x_initial[3][0])/dt, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, (self.x[4][0] - self.x_initial[4][0])/dt, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, (self.x[5][0] - self.x_initial[5][0])/dt, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, (self.x[6][0] - self.x_initial[6][0])/dt, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, (self.x[7][0] - self.x_initial[7][0])/dt, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, (self.x[8][0] - self.x_initial[8][0])/dt]])

        self.P = 500 * np.eye(self.n)                                     # covariance state matrix
        # discrete noise model, dt is small
        self.sigma_w = 1
        # E[w*w.T]
        self.Q = (self.sigma_w ^ 2) * np.dot(self.B, self.B.T)
        # E[v*v.T]
        self.R = R                                    # measurement covariance matrix

    # prediction step
    def aprioriEstimation(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)         # state vector estimation
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # state covariance estimation

    # update step
    def aposterioriEstimation(self, y):
            S = np.linalg.inv(np.dot(np.dot(self.C, self.P), self.C.T) + self.R)
            self.K = np.dot(np.dot(self.P, self.C.T), S)
            self.x = self.x + np.dot(self.K, (y - np.dot(self.C, self.x)))
            self.P = np.dot(np.eye(self.n) - np.dot(self.K, self.C), self.P)

def estimation(KF, y, u):
    state = np.empty(shape=[KF.n, 1])
    covariance = np.empty(shape=[KF.n, 1])
    for index in range(0, y.shape[0]):
        KF.aprioriEstimation(u)
        observations = np.array(y[index].reshape(y.shape[1], 1))
        KF.aposterioriEstimation(observations)
        state = np.concatenate([state, KF.x], axis=1)
        KF.x_initial = KF.x
        covariance = np.concatenate([state, KF.P], axis=1)
    state = np.delete(state, 0, axis=1)
    covariance = np.delete(covariance, 0, axis=1)
    return state, covariance
