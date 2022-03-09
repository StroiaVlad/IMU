import os
import pandas as pd

def getDatasetPath(chosenDataset, withNoise):
    # folderNames is a dict containing as keys numbers from 1 to 7 and as values the folder names of the trajectory datasets
    folderNames = {1: "TRAJECTORY 1", 2: "TRAJECTORY 2", 3:"TRAJECTORY 3", 4: "TRAJECTORY 4", 5: "TRAJECTORY 5", 6: "TRAJECTORY 6", 7: "TRAJECTORY 7"}
    # datasets is a dict containing as keys numbers from 1 till 7 and as values tuples containing the names of the datasets
    datasets = {1: ("static_100Hz", "static_100Hz_with_noise"), 2: ("straight_50Hz", "straight_50Hz_with_noise"), 3: ("car_turns_100Hz", "car_turns_100Hz_with_noise"),
                    4: ("fast_car_50Hz", "fast_car_50Hz_with_noise"), 5: ("air_turns_50Hz", "air_turns_50Hz_with_noise"),
                    6: ("climb_and_descent_50Hz", "climb_and_descent_50Hz_with_noise"), 7: ("whole_flight_20Hz", "whole_flight_20Hz_with_noise")}
    folderPath = os.getcwd() + '/measurements/' + str(folderNames[chosenDataset]) + '/'
    filePath = folderPath + str(datasets[chosenDataset][1]) if withNoise else folderPath + str(datasets[chosenDataset][0])
    return folderPath, filePath

def getDataframe(filePath):
    with open(filePath, 'r') as f: # firstly open the text file for reading (r)
        contents = f.readlines() # read all the lines of the text file and return them as a list of strings
    Fs = [] #sampling frequency
    P0 = [] #initial position of IMU
    V0 = [] #initial velocity of IMU
    E0 = [] #initial Euler Angles
    tempMeasurements = []
    indexStartMeasurements = []
    for index, line in enumerate(contents):
        if len(indexStartMeasurements) == 1:
            tempMeasurements.append(contents[index])
            continue
        if line.find("# Sampling frequency (Hz) :") >= 0: # if "# Sampling frequency (Hz) :" string is found, update Fs
            Fs.append(float(contents[index + 1]))
            continue
        if line.find("# IMU initial position :") >= 0: # if "# IMU initial position :" string is found, update P0
            P0.append(contents[index + 1])
            P0 = P0[0]
            continue
        if line.find("# IMU initial velocity :") >= 0: # if "# IMU initial velocity :" string is found, update V0
            V0.append(contents[index + 1])
            V0 = V0[0]
            continue
        if line.find("# IMU initial Euler's angles :") >= 0: # if "## IMU initial Euler's angles :" update E0
            E0.append(contents[index + 1])
            E0 = E0[0]
        if line.find("Time (GPS)") >= 0: # if "Time (GPS) :" update indexStartMeasurements
            indexStartMeasurements.append(index)
            tempMeasurements.append(contents[index])
            continue

    df = pd.read_csv(filepath_or_buffer=filePath, sep="\t", header=indexStartMeasurements[0])
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')] # drop columns in the dataframe that start with Unnamed
    df.to_csv('data.csv') # save the csv
    return Fs, P0, V0, E0, df
