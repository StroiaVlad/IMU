import os
import numpy as np
import pandas as pd

def getDatasetPath(chosenDataset, withNoise):
    """getDatasetPath function return the folderPath and filePath of the Trajectory Dataset chosen by the user via chosenDataset integer (number of dataset), withNoise flag"""
    # folderNames is a dict containing as keys numbers from 1 to 7 and as values the folder names of the trajectory datasets
    folderNames = {1: "TRAJECTORY 1", 2: "TRAJECTORY 2", 3:"TRAJECTORY 3", 4: "TRAJECTORY 4", 5: "TRAJECTORY 5", 6: "TRAJECTORY 6", 7: "TRAJECTORY 7"}
    # datasets is a dict containing as keys numbers from 1 till 7 and as values tuples containing the text files names of the datasets
    datasets = {1: ("static_100Hz", "static_100Hz_with_noise"), 2: ("straight_50Hz", "straight_50Hz_with_noise"), 3: ("car_turns_100Hz", "car_turns_100Hz_with_noise"),
                    4: ("fast_car_50Hz", "fast_car_50Hz_with_noise"), 5: ("air_turns_50Hz", "air_turns_50Hz_with_noise"),
                    6: ("climb_and_descent_50Hz", "climb_and_descent_50Hz_with_noise"), 7: ("whole_flight_20Hz", "whole_flight_20Hz_with_noise")}
    # os.getcwd() returns the current working directory(path) in which the project is found. folderPath concatenates to the current path, /measurements/
    # and then folderName of the chosen dataset by the user
    folderPath = os.getcwd() + '/measurements/' + str(folderNames[chosenDataset]) + '/'
    # the path to the text file dataset is the folderPath concatenated to the tuple returned by the datasets(which is a dictiornary) of [chosenDataset(which is a key)]
    # depending on the withNoise flag passed by the user, either the first name from the tuple is passed (withNoise is False) either the second name (withNoise is True)
    filePath = folderPath + str(datasets[chosenDataset][1]) if withNoise else folderPath + str(datasets[chosenDataset][0])
    return folderPath, filePath

def getDataframe(filePath):
    """getDataframe function return the most relevant paramenters from the Trajectory Datasets, namely Fs, P0, V0, V0, E0 and the dataframe of the actual measurements by accepting the filePath as an argument"""
    with open(filePath, 'r') as f:  # firstly open the text file found at filePath for reading (r) from it
        contents = f.readlines()    # read all the lines of the text file and return them as a list of strings stored in the contents variable
    Fs = np.zeros(shape=[1])        # sampling frequency initialized as a [0] array
    P0 = np.zeros(shape=[1, 3])     # initial position of IMU initialized as a row vector of [0, 0, 0]
    V0 = np.zeros(shape=[1, 3])     # initial velocity of IMU initialized as a row vector of [0, 0, 0]
    E0 = np.zeros(shape=[1, 3])     # initial Euler Angles initialized as a row vector of [0, 0, 0]
    indexStartMeasurements = []     # temporary dummy variable used to store the index(the line number) from the text file from which the measurements start (empty array)

    for index, line in enumerate(contents):     # iterating through the contents lists of strings while keeping track of each line index and line string array
        if len(indexStartMeasurements) == 1:    # if the indexStartMeasurements is not empty anymore (its length is 1) it means that iterator has passed the line from which the measurements start
            break                               # since the line index of the text file from which the measurements start is found, exit from the loop
        if line.find("# Sampling frequency (Hz) :") >= 0:   # if "Sampling frequency (Hz) :" string is found in the current line
            Fs[0] = float(contents[index + 1])              # update Fs by taking the float value of the contents string array found at the next index, Fs value found on the next line after Sampling frequency (Hz)
            continue
        if line.find("# IMU initial position :") >= 0:      # if "IMU initial position :" string is found in the current line
            P0[0] = eval(contents[index + 1])               # update P0 by converting via eval() the contents string array at the next index to a float array
            continue
        if line.find("# IMU initial velocity :") >= 0:      # if "IMU initial velocity :" string is found in the current line
            V0[0] = eval(contents[index + 1])               # update V0 by converting via eval() the contents string array at the next index to a float array
            continue
        if line.find("# IMU initial Euler's angles :") >= 0:    # if "IMU initial Euler's angles :" is found in the current line
            E0[0] = eval(contents[index + 1])                   # update E0 by converting via eval() the contents string array at the next index to a float array
        if line.find("Time (GPS)") >= 0:                        # if "Time (GPS) :" is found in the current line
            indexStartMeasurements.append(index)                # append to the empty array the line index of the text file from which the measurements start
            continue
    df = pd.read_csv(filepath_or_buffer=filePath, sep="\t", header=indexStartMeasurements[0])   # make a df measurements dataframe out of the text file specified via filePath and the indexStartMeasurements
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]                                       # drop the additional, redundant and unnecessary columns in the dataframe that start with Unnamed
    df.to_csv('data.csv')                                                                       # save the measurements dataframe as a csv in the project
    return Fs, P0, V0, E0, df
