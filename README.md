# Klebsiellas Antibiotic Prediction Library

## Folder structure:

- **Data folder**:
    - GM_data.pkl: data collected from _Gregorio Marañón_ Hospital and normalized by itself.
    - RyC_data.pkl: data collected from _Ramón y Cajal_ Hospital and normalized by itself.
    - GM_data_both.pkl: data collected from _Gregorio Marañón_ Hospital and normalized by both Hospital's data.
    - RyC_data_both.pkl: data collected from _Ramón y Cajal_ Hospital and normalized by both Hospital's data.
- **Results**: pkl files with the SSHIBA model trained for each case.
- **ksshiba**: SSHIBA library model.
    
## Preprocess_data:
The _preprocess_data_ class process data from both hospitals. The data is normalized first
by Total Ion Count (TIC) normalization and then by StandardScaler. The user can choose between
normalizing the data mixed or separated by hospital. Regarding the Y, only the antibiotics
present in both hospital data are kept. 

Some variables can be choosen to: keep only common antibiotics, return one or another hospital data 
or normalize both data jointly.
````python
# Load the method
import preprocess_data as pp
# Define the way do you want to preprocess and load data.
# Path to the data folder
# Load both data
gm_path = "./data/old_data/"
ryc_path = "./data/Klebsiellas_RyC/"
# Total number of antibiotics present in GM data:
gm_n_labels = 18

# Process the data
gm_data, ryc_full_data = pp.process_hospital_data(gm_path=gm_path, gm_n_labels=gm_n_labels, ryc_path=ryc_path,
                                                  keep_common=True, return_GM=True, return_RYC=True, norm_both=True)
````

This class also contains more ways to preprocess the data. Take a look to them in case you need them.

## Run baselines
The class run_baselines.py basically runs all the baselines proposed in this project. By default it runs:
- Logitic Regressor:
    - With L1 regularization.
    - With L2 regularization.
    - With Elastic-Net regularization.
- Gaussian Process Classifier: just with a cross-validation of the kernel used.

## SSHIBA model:
The model proposed to learn the data is **SSHIBA** [[1]](#1). There are 3 scripts that run this model
in the three scenarios we have proposed:
* GM_model.py: learn a SSHIBA model focused on GM data.
* RYC_model.py: learn a SSHIBA model focused on RyC data.
* both_model.py: learn a SSHIBA model combining both data. 

## Results:
The _show_results_ is a script used to analyze de results of the different SSHIBA models proposed.

## Miscellanea:

All scripts not present in previous sections are miscellanea used to simplify/help the other main scripts.


## References
<a id="1">[1]</a>
Sevilla-Salcedo, Carlos, Vanessa Gómez-Verdejo, and Pablo M. Olmos. 
"Sparse Semi-supervised Heterogeneous Interbattery Bayesian Analysis." 
arXiv preprint arXiv:2001.08975 (2020).