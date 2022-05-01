import os
import numpy as np
import pandas as pd
from string2Numpy import stringList2Numpy

def getDatasetPath(chosenDataset, withNoise, withGNSS):
    """getDatasetPath function return the folderPath and filePath of the Trajectory Dataset chosen by the user via chosenDataset integer (number of dataset), withNoise flag"""
    # folderNames is a dict containing as keys numbers from 1 to 7 and as values the folder names of the trajectory datasets
    folderNames = {1: "TRAJECTORY 1", 2: "TRAJECTORY 2", 3:"TRAJECTORY 3", 4: "TRAJECTORY 4",
                   5: "TRAJECTORY 5", 6: "TRAJECTORY 6", 7: "TRAJECTORY 7", 8: "TRAJECTORY 8", 9: "TRAJECTORY 9"}
    # datasets is a dict containing as keys numbers from 1 till 7 and as values tuples containing the text files names of the datasets
    datasets = {1: ("static_100Hz", "static_100Hz_with_noise"), 2: ("straight_50Hz", "straight_50Hz_with_noise"), 3: ("car_turns_50Hz.imu", "car_turns_50Hz_with_noise.imu"),
                4: ("fast_car_50Hz.imu", "fast_car_50Hz_with_noise.imu"), 5: ("air_turns_50Hz.imu", "air_turns_50Hz_with_noise.imu"),
                6: ("climb_and_descent_50Hz.imu", "climb_and_descent_50Hz_with_noise.imu"), 7: ("flight_50Hz.imu", "flight_50Hz_with_noise.imu"),
                8: ("air_rolls_and_straight_50hz.imu", "air_rolls_and_straight_50Hz_with_noise.imu", "air_rolls_and_straight_gnss_pos.txt"),
                9: ("circular.imu", "circular_with_noise.imu", "circular_no_noise_gnss_pos.txt", "circular_gnss_pos.txt")}
    # os.getcwd() returns the current working directory(path) in which the project is found. folderPath concatenates to the current path, /measurements/
    # and then folderName of the chosen dataset by the user
    folderPath = os.getcwd() + '/measurements/' + str(folderNames[chosenDataset]) + '/'
    # the path to the text file dataset is the folderPath concatenated to the tuple returned by the datasets(which is a dictiornary) of [chosenDataset(which is a key)]
    # depending on the withNoise flag passed by the user, either the first name from the tuple is passed (withNoise is False) either the second name (withNoise is True)
    filePathIMU = folderPath + str(datasets[chosenDataset][1]) if withNoise else folderPath + str(datasets[chosenDataset][0])
    filePathGNSS = folderPath + str(datasets[chosenDataset][2]) if withGNSS and (chosenDataset == 8 or chosenDataset == 9) else None
    return folderPath, filePathIMU, filePathGNSS

def getDataframeIMU(filePathIMU):
    """getDataframe function return the most relevant paramenters from the Trajectory IMU Datasets, namely Fs, P0, V0, V0, E0 and the dataframe of the actual measurements by accepting the filePath as an argument"""
    with open(filePathIMU, 'r') as f:  # firstly open the text file found at filePath for reading (r) from it
        contents = f.readlines()    # read all the lines of the text file and return them as a list of strings stored in the contents variable
    Fs_IMU = np.zeros(shape=[1])        # sampling frequency initialized as a [0] array
    P0 = np.zeros(shape=[1, 3])     # initial position of IMU initialized as a row vector of [0, 0, 0]
    V0 = np.zeros(shape=[1, 3])     # initial velocity of IMU initialized as a row vector of [0, 0, 0]
    E0 = np.zeros(shape=[1, 3])     # initial Euler Angles initialized as a row vector of [0, 0, 0]
    indexStartMeasurements = []     # temporary dummy variable used to store the index(the line number) from the text file from which the measurements start (empty array)

    for index, line in enumerate(contents):     # iterating through the contents lists of strings while keeping track of each line index and line string array
        if len(indexStartMeasurements) == 1:    # if the indexStartMeasurements is not empty anymore (its length is 1) it means that iterator has passed the line from which the measurements start
            break                               # since the line index of the text file from which the measurements start is found, exit from the loop
        if line.find("# Sampling frequency (Hz) :") >= 0:   # if "Sampling frequency (Hz) :" string is found in the current line
            Fs_IMU[0] = float(contents[index + 1])              # update Fs by taking the float value of the contents string array found at the next index, Fs value found on the next line after Sampling frequency (Hz)
            continue
        if line.find("# IMU initial position :") >= 0:      # if "IMU initial position :" string is found in the current line
            P0[0] = stringList2Numpy(contents[index + 1])
            #P0[0] = eval(contents[index + 1])               # update P0 by converting via eval() the contents string array at the next index to a float array
            #P0[0] = [6378137, 0, 0]
            continue
        if line.find("# IMU initial velocity :") >= 0:      # if "IMU initial velocity :" string is found in the current line
            V0[0] = stringList2Numpy(contents[index + 1])
            #V0[0] = eval(contents[index + 1])               # update V0 by converting via eval() the contents string array at the next index to a float array
            #V0[0] = [0, 0, 0]
            continue
        if line.find("# IMU initial Euler's angles :") >= 0:    # if "IMU initial Euler's angles :" is found in the current line
            E0[0] = stringList2Numpy(contents[index + 1])
            #E0[0] = eval(contents[index + 1])                   # update E0 by converting via eval() the contents string array at the next index to a float array
            #E0[0] = [0, 0, 0]
        if line.find("Time (GPS)") >= 0:                        # if "Time (GPS) :" is found in the current line
            indexStartMeasurements.append(index)                # append to the empty array the line index of the text file from which the measurements start
            continue
    df_IMU = pd.read_csv(filepath_or_buffer=filePathIMU, sep="\t", header=indexStartMeasurements[0])   # make a df measurements dataframe out of the text file specified via filePath and the indexStartMeasurements
    df_IMU = df_IMU.loc[:, ~df_IMU.columns.str.startswith('Unnamed')]                                       # drop the additional, redundant and unnecessary columns in the dataframe that start with Unnamed
    df_IMU.to_csv('data.csv')                                                                       # save the measurements dataframe as a csv in the project
    return Fs_IMU, P0, V0, E0, df_IMU

def getDataframeGNSS(filePathGNSS):
    """getDataframe function return the most relevant paramenters from the Trajectory Datasets, namely X, Y, Z, Vx, Vy, Vz, Xcov, Ycov,Zcov, Vxcox, Vycov, Vzcov"""
    if filePathGNSS == None:
        Fs_GNSS = None
        df_GNSS = None
        return Fs_GNSS, df_GNSS

    with open(filePathGNSS, 'r') as f:  # firstly open the text file found at filePath for reading (r) from it
        contents = f.readlines()    # read all the lines of the text file and return them as a list of strings stored in the contents variable
    Fs_GNSS = np.zeros(shape=[1])        # sampling frequency initialized as a [0] array
    indexStartMeasurements = []     # temporary dummy variable used to store the index(the line number) from the text file from which the measurements start (empty array)

    for index, line in enumerate(contents):     # iterating through the contents lists of strings while keeping track of each line index and line string array
        if len(indexStartMeasurements) == 1:    # if the indexStartMeasurements is not empty anymore (its length is 1) it means that iterator has passed the line from which the measurements start
            break                               # since the line index of the text file from which the measurements start is found, exit from the loop
        if line.find("# Sampling frequency (Hz) :") >= 0:   # if "Sampling frequency (Hz) :" string is found in the current line
            Fs_GNSS[0] = float(contents[index + 1])              # update Fs by taking the float value of the contents string array found at the next index, Fs value found on the next line after Sampling frequency (Hz)
            continue
        if line.find("Time (GPS)") >= 0:                        # if "Time (GPS) :" is found in the current line
            indexStartMeasurements.append(index)                # append to the empty array the line index of the text file from which the measurements start
            continue
    df_GNSS = pd.read_csv(filepath_or_buffer=filePathGNSS, sep="\t", header=indexStartMeasurements[0])   # make a df measurements dataframe out of the text file specified via filePath and the indexStartMeasurements
    df_GNSS = df_GNSS.loc[:, ~df_GNSS.columns.str.startswith('Unnamed')]                                       # drop the additional, redundant and unnecessary columns in the dataframe that start with Unnamed
    df_GNSS.to_csv('data.csv')                                                                       # save the measurements dataframe as a csv in the project
    return Fs_GNSS, df_GNSS
