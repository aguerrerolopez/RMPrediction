import pickle
import sys
import numpy as np
sys.path.insert(0, "./lib")
import json
import telegram
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score as auc
from matplotlib import pyplot as plt

##################### PARAMETERS SELECTION ####################3
# Which baseline do you want to use:
randomforest = 0
knn=0
svm=1
gp=0

########################## LOAD DATA ######################
folds_path = "./data/RYC_5STRATIFIEDfolds_paper.pkl"
data_path = "./data/ryc_data_paper.pkl"
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)
with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)

###### PRED CARB BLEE
fen = gm_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']].dropna()
maldi = gm_data['maldi'].loc[fen.index]

##################### TRAIN BASELINES AND PREDICT ######################
results = np.zeros((5,3))

for f in range(len(folds["train"])):
    print("Training fold: ", f)
    
    if "BTSE3" in folds["train"][f]: folds["train"][f].remove("BTSE3")
    if "BTSH11" in folds["train"][f]: folds["train"][f].remove("BTSH11")
    if "BTSE3" in folds["val"][f]: folds["train"][f].remove("BTSE3")
    if "BTSH11" in folds["val"][f]: folds["train"][f].remove("BTSH11")

    x0_tr, x0_val = maldi.loc[folds["train"][f]], maldi.loc[folds["val"][f]]
    y_tr, y_val = np.vstack(fen.loc[folds["train"][f]].values), np.vstack(fen.loc[folds["val"][f]].values)

    x0_tr= np.vstack(x0_tr.values).astype(float)
    mean = np.mean(x0_tr)
    x0_tr /= mean
    x0_val = np.vstack(x0_val.values).astype(float)
    x0_val /= mean

    if randomforest:
        from sklearn.ensemble import RandomForestClassifier as RFC
        clf = RFC(n_jobs=-1)
        param_grid = {'n_estimators': [50, 100, 150],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'criterion' :['gini']}
        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    elif knn:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_jobs=-1)
        param_grid = {'n_neighbors': range(1,20)}
        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    elif svm:
        from sklearn.svm import SVC
        clf = SVC(probability=True, kernel="rbf")
        param_grid = {'C': [0.01, 0.1 , 1, 10]}
        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    elif gp:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import DotProduct
        # IF YOU WANT LINEAR KERNEL
        kernel= DotProduct()
        CV_rfc = GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
        # IF YOU WANT RBF KERNEL LET IT BE DEFAULT
        # CV_rfc = GaussianProcessClassifier(n_jobs=-1)

    if randomforest:
        for c in range(y_tr.shape[1]):
            CV_rfc.fit(x0_tr, y_tr[:, c])
            y_pred = CV_rfc.predict_proba(x0_val)[:, 1]
            score= auc(y_val[:, c], y_pred)
            results[f,c]=score
    
    elif svm:
        for c in range(y_tr.shape[1]):
            print("TRAINING SVM")
            # USE LAST PREDICTION TO TRAIN NEXT TASK (forcing to exploit the correlation)
            # if c>0:
            #     x0_tr = np.hstack((x0_tr, y_tr[:, c-1][:, np.newaxis]))
            #     x0_val = np.hstack((x0_val, y_pred[:, np.newaxis]))
            CV_rfc.fit(x0_tr, y_tr[:, c])
            y_pred = CV_rfc.predict_proba(x0_val)[:, 1]
            score= auc(y_val[:, c], y_pred)
            results[f,c]=score
    else:
        for c in range(y_tr.shape[1]):
            CV_rfc.fit(x0_tr, y_tr[:, c])
            y_pred = CV_rfc.predict_proba(x0_val)
            y_pred = y_pred[:,1]
            score= auc(y_val[:, c], y_pred)
            results[f,c]=score

    print(results)

print("Results")
print(np.mean(results, axis=0))
print(np.std(results, axis=0))