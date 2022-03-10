import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from getMeasurements import getDatasetPath, getDataframe

chosenDataset = 1 #specify which Trajectory dataset to use
withNoise = True #add noise to it

# Press the green button in the gutter to run the script
if __name__ == '__main__':
    folderPath, filePath = getDatasetPath(chosenDataset, withNoise)
    print(folderPath, filePath)
    Fs, P0, V0, E0, y = getDataframe(filePath)
    print(y.head(), len(y))
