# Klebsiellas Antibiotic Prediction Library

## Folder structure:

- **Data folder**:
    - GM_data: data collected from _Gregorio Marañón_ Hospital
    - RyC_data: data collected from _Ramón y Cajal_ Hospital.
- **Results**: pkl files with the scores of each model trained will be stored here.
    - Plots: plots of the results will be stored here.
    
## Preprocess:
The preprocess class contains all the methods to read and preprocess the data. It can process only the
spectra data, only the antibiotic resistance or both at the same time. A common way to use it is:
````python
# Load data
import preprocess as pp
# Define the way do you want to preprocess and load data.
# Path to the data folder
data_path = "./data/old_data/"
# Format on which the data is stored
format = "zip"
# Boolean to know if you want to drop some columns on the antibiotics resistance.
drop = True
# List of columns to drop if the boolean is set to true.
columns_drop = ["AMPICILINA"]
# Boolean to know if you wanna to impute missing values in antibiotic resistance columns.
impute = True
# List with the limits to cut off the spectra
limits = [2005, 19805]
# Number of antibiotic resistance columns that exists on our dataset.
n_labels = 18


# Load data
data = pp.load_data(path=data_path, format=format)

# Preprocess data
x, x_axis, y, labels_name = pp.prepare_data(data, drop=drop, columns_drop=columns_drop,
                                            impute=impute, limits=limits, n_labels=n_labels)
````

This class also contains more ways to preprocess the data. Take a look to them in case you need them.

## Run baselines
The class run_baselines.py basically runs all the baselines proposed in this project. By default it runs:
- Logitic Regressor:
    - With L1 regularization.
    - With L2 regularization.
    - With Elastic-Net regularization.
- Gaussian Process Classifier: just with a cross-validation of the kernel used.