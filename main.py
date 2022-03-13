import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from getMeasurements import getDatasetPath, getDataframe
from KalmanFilter import *

chosenDataset = 1 #specify which Trajectory dataset to use
withNoise = True #add noise to it

# Press the green button in the gutter to run the script
if __name__ == '__main__':
    folderPath, filePath = getDatasetPath(chosenDataset, withNoise)
    print(folderPath, filePath)
    Fs, P0, V0, E0, y = getDataframe(filePath)

    dt = 1/Fs[0]
    u = 0.25 * np.ones([6, 1])
    R = y.iloc[:, 1:].cov()
    observations = y.iloc[:, 1:] # discard time GPS from the dataframe

    obs_numpy = observations.to_numpy()

    KF = KalmanFilter(P0, V0, E0, R, dt)
    state, covariance = estimation(KF, obs_numpy, u)


