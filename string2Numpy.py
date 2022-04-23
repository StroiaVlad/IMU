import numpy as np
def stringList2Numpy(line):
    stringDigits = "" # string that concatenates the digits appearing in the line
    digitCount = 0  # counts the digits in the line
    arrayDigits = [] # array that will contain the concatenated stringDigits
    for c in line: # iterating through each character of the string
        if c.isdigit() or c == ".": # if the character is a digit or it is a .
            stringDigits = stringDigits + c # concatenate to stringDigits the character c
            digitCount = digitCount + 1 # increment the digitCount
        if c == " " or c == "]": # if the c is either a " " or a "]"
            if digitCount !=0 : # if digitCount is different from 0, meaning that it is the first black space after a digit
                arrayDigits.append(stringDigits) # then append the concatenated stringDigits to the arrayDigits
            digitCount = 0 # set the digitCount to 0, since the counter has to be restarted for the next sequence of digits in the line
            stringDigits = "" # reset the stringDigits to an empty string
    return np.asarray(arrayDigits, dtype=float) # return a numpy array of a arrayDigits
