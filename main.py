from getMeasurements import getDatasetPath, getDataframeIMU, getDataframeGNSS
from KalmanFilter import *
from conversion import *
from results import *

# User specified chosenDataset and withNoise flag
chosenDataset = 2 # specify which Trajectory dataset to use
withNoise = False  # Do you want noise to your data or not?
withGNSS = False # Do you want to hybridize with GNSS data?

# Press the green button in the gutter to run the script
if __name__ == '__main__':
    folderPath, filePathIMU, filePathGNSS = getDatasetPath(chosenDataset, withNoise, withGNSS)  # get the folderPath and filePath of the chosen Trajectory dataset
    print(folderPath, filePathIMU)  # check whether the folderPath and filePath are what they should be
    Fs_IMU, P0, V0, E0, u = getDataframeIMU(filePathIMU)  # get the most relevant parameters out of the text file, note that u is the dataframe of measurements
    dt = 1 / Fs_IMU[0]  # the time increment used for Kalman Filtering apriori estimation
    print(u.shape)  # check if the measurements dataframe is the intended shape
    measurements = u.iloc[:, 1:]  # measurements is a new DataFrame which contains all the rows and columns of u, but except the "Time GPS" columns
    measurementsNumpied = measurements.to_numpy()  # convert the measurements DataFrame into a numpy array
    GPS_time_IMU = u.iloc[:, 0]  # GPS time from the IMU dataset

    Fs_GNSS, y = getDataframeGNSS(filePathGNSS)  # get the most relevant parameters out of the text file, note that y is the dataframe of measurements
    observations, observationsNumpied, GPS_time_GNSS = None, None, None
    if bool(Fs_GNSS):
        print(y.shape)  # check if the measurements dataframe is the intended shape
        observations = y.iloc[:, 1:]  # observations is a new DataFrame which contains all the rows and columns of y, but except the "Time GPS" columns
        observationsNumpied = observations.to_numpy()  # convert the observations DataFrame into a numpy array
        GPS_time_GNSS = y.iloc[:, 0]  # GPS time from the GNSS dataset

    R = u.iloc[:, 1:].cov()  # measurements covariance matrix
    KF = KalmanFilter(P0, V0, E0, R, dt)  # Instantiating the KF object that implements the apriori and aposteriori steps of Kalman Filtering
    state = KF.estimation(KF, measurementsNumpied, GPS_time_IMU, dt, withGNSS, observationsNumpied, GPS_time_GNSS)  # estimate the state vector and covariance matrix from the Kalman Filter
    print(state[0, :])                                                            # print the state

    T = np.arange(0, u.shape[0] / Fs_IMU[0], 1 / Fs_IMU[0])
    #print(T.shape)
    plotPosition(T, state)
    plotVelocity(T, state)
    plotAttitude(T, state)
    plotProfile(T, state)