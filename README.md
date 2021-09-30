# Library that implements "Development and Validation of a MALDI-TOF-Based Model to Predict Extended-Spectrum Beta-Lactamase and/or Carbapenemase-Producing Klebsiella pneumoniae Clinical Isolates" paper


## Folder structure:

- **Data folder**:
    - GM/GM_AST.xlsx: AST associated with each MALDI from _Gregorio Marañón_ Hospital (GM)
    - RyC/RyC_AST:  AST associated with each MALDI from _Ramón y Cajal_ Hospital (RyC).
    - gm_data_paper.pkl: MALDI MS and AST stored and preprocessed from GM.
    - ryc_data_paper.pkl: MALDI MS and AST stored and preprocessed from RyC.
    - GM_XSTRATIFIEDfolds_paper.pkl: X(5 or 10) train-test fold partition for GM data.
    - GM_XSTRATIFIEDfolds_paper.pkl: X(5 or 10) train-test fold partition for RyC data.
- **lib**: SSHIBA library model.

## Preprocess the data:
We have two preprocess data scripts:
- data_hgm.py
- data_hgm.py

Every scripts does the same. First, we read the data of the 3 days of the hospital. For each unique sample we have it repetead several times. To select a MALDI MS for strain we propose the median approach: for each unique sample we calculate the median synthethic sample and then we make the difference between all the real samples and our median one. The sample that is closest to our median is the one that we are going to use to train our model. In that way we get rid of possible outliers and measurement errors. Finally, we propose to make 10 stratified folds train-test our results. The MALDIs signal is then normalized by TIC technique. The preprocess pipeline can be seen in Figure 1.
![alt text](images_readme/preprocess.png)

## SSHIBA model:
The model used to learn the data is Sparse Semi-supervised Heterogeneous Interbattery Bayesian Analysis (**SSHIBA**) [[1]](#1) and its kernel approach kernelized-SSHIBA (**KSSHIBA**) [[2]](#2). SSHIBA is a Bayesian kernel model capable of jointly working with heterogeneous multi-view data such as continuous kernelized data or multilabel, as well as dealing with missing data in any view. For this work, two scenarios are found.

### Intrahospital scenario:
Still creating the plots... It will be here soon ;)

### Interhospital scenario:
 The interhospital scenario is presented in Figure 2.
![alt text](images_readme/interhospital.png)
This scenario works with three views:
* <img src="https://render.githubusercontent.com/render/math?math=k_{n,:}^{m}">: MALDI MS kernelized (linear, rbf or pike) view.
* <img src="https://render.githubusercontent.com/render/math?math=t_{n,:}^{1}">: Hospital Origin (GM or RyC) view.
* <img src="https://render.githubusercontent.com/render/math?math=t_{n,:}^{2}">: Antibiotic resistance (CP+ESBL, ESBL, S) view

In this scenario we exploit the multiview framework of KSSHIBA and it is the major contribution of the research. Using this multiview approach, we can work with an extra label that indicates to the model from which data distribution is the sample comming from. This way, the model can learn the common information of both distributions (as they both explain the same microorganism) but getting rid of possible misinformation that can occur only in intrahospital scenarios. As it is explained on the Discussion part of the paper, this extra view helps the model to not get biased by unbalanced presence in both distributions.

The script is found in:
* interhospital_ksshiba.py

## Results:


## Latent space:
SSHIBA calculates a <img src="https://render.githubusercontent.com/render/math?math=W^{(m)}"> projection matrix associated to each view. However, when the view is kernelized it calculates the dual space projection matrix <img src="https://render.githubusercontent.com/render/math?math=A^{(m)}">, we have to recover the primal space <img src="https://render.githubusercontent.com/render/math?math=W^{(m)}"> projection matrix by doing: <img src="https://render.githubusercontent.com/render/math?math=W^{(m)} = X^{(m)T}A^{(m)}">. 
This <img src="https://render.githubusercontent.com/render/math?math=W^{(m)}"> is a projection matrix of shape _DxK_ being _D_ the dimension of the data and _K_ the dimension of the latent space that SSHIBA decides by optimisation. Then, each _K_ column of the <img src="https://render.githubusercontent.com/render/math?math=W^{(m)}"> matrix is a projection weight vector associated to each dimension data point and can be interpretable. First, we can plot each the K columns by computing the mean over the D rows to show how every view is correlated to all the others views as we can see in Figure 2:
![alt text](images_readme/bothhospital_latentspace.png)

This example is extracted from the intrahospital scenario shown in the paper.

Further explanation comming soon...

All these calculus and plots explained in this section can be found in:
* show_results.py

## References
<a id="1">[1]</a>
Sevilla-Salcedo, Carlos, Vanessa Gómez-Verdejo, and Pablo M. Olmos. 
"Sparse Semi-supervised Heterogeneous Interbattery Bayesian Analysis." 
Patern Recognition (2021).

<a id="2">[2]</a>
Sevilla-Salcedo, C., Guerrero-López, A., Olmos, P. M., & Gómez-Verdejo, V. (2020). 
Bayesian Sparse Factor Analysis with Kernelized Observations. 
arXiv preprint arXiv:2006.00968.


