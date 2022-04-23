from getMeasurements import getDatasetPath, getDataframe
from KalmanFilter import *
from conversion import *
from results import *

# User specified chosenDataset and withNoise flag
chosenDataset = 9 # specify which Trajectory dataset to use
withNoise = False  # Do you want noise to your data or not?
withGNSS = False # Do you want to hybridize with GNSS data?

# Press the green button in the gutter to run the script
if __name__ == '__main__':
    folderPath, filePath = getDatasetPath(chosenDataset, withNoise)  # get the folderPath and filePath of the chosen Trajectory dataset
    print(folderPath, filePath)  # check whether the folderPath and filePath are what they should be
    Fs, P0, V0, E0, y = getDataframe(filePath)  # get the most relevant parameters out of the text file, note that y is the dataframe of measurements
    dt = 1 / Fs[0]  # the time increment used for Kalman Filtering apriori estimation
    print(y.shape)  # check if the measurements dataframe is the intended shape
    measurements = y.iloc[:, 1:]  # observations is a new DataFrame which contains all the rows and columns of y, but except the "Time GPS" columns
    measurementsNumpied = measurements.to_numpy()  # convert the observations DataFrame into a numpy array
    GPS_time = y.iloc[:, 0]  # GPS time fron the dataset

    observations = 0 * np.ones([measurementsNumpied.shape[1], 1])  # initializing the input command vector as a 6x1 numpy array
    # rot_matrix = ECEF2body(u[0:3, :], E0[0][0], E0[0][1], E0[0][2]) # rotation matrix from ECEF to BODY
    # rot_matrix = np.array(list(rot_matrix[:, :]), dtype=np.float64) # making a numpy array
    # u[0:3, :] = np.dot(np.linalg.inv(rot_matrix), u[0:3, :]) # converting the Specific force from BODY frame to ECEF frame
    # u[2, :] = u[2, :] + 9.809792 # removing the gravity component from the accelerometer data
    # u[3:, :] = ECEF2ECEF0(u[3:, :], GPS_time[0]) # converting the gyro data from ECEF0 to ECEF

    R = y.iloc[:, 1:].cov()  # measurements covariance matrix
    KF = KalmanFilter(P0, V0, E0, R, dt)  # Instantiating the KF object that implements the apriori and aposteriori steps of Kalman Filtering
    state = KF.estimation(KF, measurementsNumpied, GPS_time, observations, dt)  # estimate the state vector and covariance matrix from the Kalman Filter
    #print(state)                                                            # print the state

    T = np.arange(0, y.shape[0] / Fs[0], 1 / Fs[0])
    #print(T.shape)
    plotPosition(T, state)
    plotVelocity(T, state)
    plotAttitude(T, state)
    plotProfile(T, state)