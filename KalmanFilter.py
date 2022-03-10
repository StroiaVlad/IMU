import numpy as np

class KalmanFilter:
    def __init__(self, P0, V0, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.x = np.array([P0.T, V0.T])     # state vector X
        self.n = np.shape(A)[0]             # row dimension of the A matrix
        self.P = np.eye(self.n)             # covariance state matrix
        self.Q = np.eye(self.n)
        self.R = np.eye(self.n)
        self.K = 0                          # initializing the Kalman Gain to 1

    def aprioriEstimation(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)         # state vector estimation
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # state covariance estimation

    def aposterioriEstimation(self, y):
            S = np.linalg.inv(np.dot(np.dot(self.C, self.P), self.C.T) + self.R)
            self.K = np.dot(np.dot(self.P, self.C.T), S)
            self.x = self.x + np.dot(self.K, (y - np.dot(self.C, self.x)))
            self.P = np.dot(np.eye(self.n) - np.dot(self.K, self.C), self.P)

def estimation(self, KF, y):
    state = np.empty(shape=[KF.n, 1])
    covariance = np.empty(shape=[KF.n, 1])
    for yk in y:
        KF.aprioriEstimation(0)
        KF.aposterioriEstimation(yk)
        state = KF.x
        #covariance[k] = KF.P

dt = 0.1
A = np.array([1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0, 0],
             [0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0],
             [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt],
             [0, 0, 0, 1, 0, 0, dt, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, dt, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, dt],
             [0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1])
