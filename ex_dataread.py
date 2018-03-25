import os
import re

import numpy as np
import pandas as pd
from algo import __trainSVM
from algo import manipulate_data
# pandas.read_excel() depends on xlrd package
try:
    import xlrd
except ImportError:
    print("Please install xlrd: pip install xlrd")
    exit(1)

# Change to current directory, so that we can use relative paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Regex stuff (complicated, don't worry). We use these to increase performance of the file name parsing
name_reg = re.compile(r'([\d]*)-(0\.[0-9]*)\.[xls]')  # catces "50-0.01.xls" file name formatting
name_reg2 = re.compile(r'([\d]*)_(\.[0-9]*)\.[xls]')  # catches "450_.01.xls" file name formatting

# Data file path (it is a relative path, not an absolute one)
data_path = 'data/'

# A list to contain all Excel data
all_data = []

# Read all files and folders inside "data" directory
data_folders = sorted(os.listdir(data_path))  # sorted function sorts in ascending order

# Loop through all folders, like "50C" or "650C".
# Here, we are NOT checking for links, files, only assuming that we only have directories
for df in data_folders:
    # Now we are two levels inside, like "data/50C". Reads all files inside that directory
    data_files = sorted(os.listdir(data_path + df))

    # Now we are on the 3rd level, like "data/50C/50-0.1.xls"
    for dfile in data_files:
        #
        # PARSE FILE NAME
        #

        # Parse the file name to obtain information on the temp and the strain rate
        experiment_info = name_reg.search(dfile)
        if not experiment_info:
            experiment_info = name_reg2.search(dfile)
        # experiment_info[0] == something like "50-0.0001.x", no "ls" letters
        temp = int(experiment_info[1])  # For a file name "xxx-yyyy.xls", returns "xxx" section
        strain_rate = float(experiment_info[2])  # For a file name "xxx-yyyy.xls", returns "yyyy" section

        #
        # READ EXCEL FILE
        #

        # Column index starts from 0: col[6] == True Strain, col[7] == True Stress
        pdxls = pd.read_excel(data_path + df + '/' + dfile, dtype=np.float, usecols=(6, 7), names=('Strain', 'Stress'))

        #
        # HOW TO EXTRACT DATA FROM PANDAS OBJECT
        #

        # Method 1: Use headers
        #
        # Pandas will automatically generate an associative array in the excel file has header row(s)
        # however, the headers in the files are not very well organized to use the associative array structure
        # > true_strain = pdxls["True Strain"].values
        # > true_stress = pdxls["True Stress"].values
        # Therefore, we use "names" parameter of read_excel function to get rid of this problem
        true_strain = pdxls["Strain"].values  # export column as a numpy array
        true_stress = pdxls["Stress"].values

        # Method 2: Use slicing
        data = pdxls.values  # Export the data as a numpy array
        col1 = data[:, 0]  # first column
        col2 = data[:, 1]  # second column

        #
        # CREATE VECTORS OF TEMP AND STRAIN RATE
        #

        temp_arr = np.repeat(temp, true_strain.shape[0])
        strain_rate_arr = np.repeat(strain_rate, true_strain.shape[0])

        #
        # ORGANIZE DATA
        #

        # The structure is a numpy array and organized like this: (temp, strain rate, true strain, true stress)
        elem = np.array([temp_arr, strain_rate_arr, col1, col2])
        all_data.append(elem)
input_data_svm = reduced_size(all_data, 1)
trainSVM_temp(input_data_svm)
trainSVM_srRate(input_data_svm)
pass
