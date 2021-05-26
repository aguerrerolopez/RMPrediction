# Klebsiellas Antibiotic Prediction Library

## Folder structure:

- **Data folder**:
    - Reproducibilidad: data collected during 3 days of the same Klebsiellas from _Gregorio Marañón_ Hospital (HGM) and normalized by TIC[[1]](#1) normalization.
    - Klebsiellas_RyC: data collected during 3 days of the same Klebsiellas from _Ramón y Cajal_ Hospital (RyC) and normalized by TIC normalization.
    - DB_conjunta.xlsx: fenotype, genotype and antibiotic resistance data from both hospitals.
- **Results**: pkl files with the SSHIBA model trained for each case.
- **lib**: SSHIBA library model.
    
## Preprocess the data:
We have three preprocess data scripts (in the future will be in only one common script):
    - datos_ambos.py
    - datos_hgm.py
    - datos_ryc.py

Every scripts does the same. First, we read the data of the 3 days of the hospital. For each unique sample we have it repetead several times, this amount of times can move between 1 and 12 times. To balance the data we propose the median aproach: for each unique sample we calculate the median synthethic sample and then we make the difference between all the real samples and our median one. The sample that is closest to our median is the one that we are going to use to train our model. In that way we get rid of possible outliers and measurement errors.

Once we have this sample, we proposed to train 7 different models one per family of antibiotics. It may occur that for a specific family we have heavy unbalanced data such as AMOXICILINA which in _HGM_ we have 83 non-resistan samples and 211 resistant ones. Or the IMIPENEM case where we have 95 non-resistan and only 7 resistant samples. This can lay into an overrepresentation of the predominant label into the training phase. To tackle with that we proposed to sample randomly the underrepresented label to have the same amount of data.

After all this we propose to make 10 and 5 folds splits to then test our results.

The MALDIs signal is only normalized by TIC technique.

````python
# poner un ejemplo aquí en el futuro
````

## Run baselines
The scripts gm_baselines_byfamily and ryc_baselines_byfamily basically runs all the baselines proposed in this project. We proposed to compare ourselves with:
- SVMs: RBF and linear 5 and 10 fold.
- Random Forest: 5 and 10 fold.
- KNN: 5 and 10 fold.

## SSHIBA model:
The model proposed to learn the data is **SSHIBA** [[2]](#2)[[3]](#3). 
* First approach, 1 unique model:
    - Input views:
        - MALDI linear/rbf kernel.
        - Fenotype multilabel.
        - Genotype multilabel.
    - Output views:
        - 7 multilabel view one per antibiotic family.

For this approach we have 3 scripts:
* GM_model.py: learn a SSHIBA model focused on GM data.
* RYC_model.py: learn a SSHIBA model focused on RyC data.
* both_model.py: learn a SSHIBA model combining both data. 

* Second approach, 7 different models:
    - Input views:
        - MALDI linear/rbf kernel.
        - Fenotype multilabel.
        - Genotype multilabel.
    - Output views:
        - Family X:  multilabel view of an antibiotic family.3

For this approach we have 3 scrippts
For this approach we have 3 scripts:
* GM_model_byfamily.py: learn a SSHIBA model focused on GM data.
* RYC_model_byfamily.py: learn a SSHIBA model focused on RyC data.
* both_model_byfamily.py: learn a SSHIBA model combining both data. 

## Results:
The _show_results_ and  _show_results_byfamily_ are scripts used to analyze the results of the different SSHIBA models proposed and plot different things such as:

- AUC per antibiotic
- Common and private latent space per view
- W primal space matrix
- A dual space matrix
- Z latent space projection

## Miscellanea:

All scripts not present in previous sections are miscellanea used to simplify/help the other main scripts.


## References
<a id="1">[1]</a>
Deininger, S.O., et al. (2011) 
Normalization in MALDI-TOF Imaging Datasets of Proteins: Practical Considerations. 
Analytical and Bioanalytical Chemistry, 401, 167-181.
https://doi.org/10.1007/s00216-011-4929-z

<a id="2">[2]</a>
Sevilla-Salcedo, Carlos, Vanessa Gómez-Verdejo, and Pablo M. Olmos. 
"Sparse Semi-supervised Heterogeneous Interbattery Bayesian Analysis." 
arXiv preprint arXiv:2001.08975 (2020).

<a id="3">[3]</a>
Sevilla-Salcedo, C., Guerrero-López, A., Olmos, P. M., & Gómez-Verdejo, V. (2020). 
Bayesian Sparse Factor Analysis with Kernelized Observations. 
arXiv preprint arXiv:2006.00968.


