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
svm=0
gp=1

########################## LOAD DATA ######################
folds_path = "./data/RYC_5STRATIFIEDfolds_muestrascompensada_experimentAug.pkl"
data_path = "./data/ryc_data_expAug.pkl"
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)
with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)

###### PRED CARB BLEE
fen = gm_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']].dropna()

maldi = gm_data['maldi'].loc[fen.index]
ab = gm_data['full'][['AMOXI/CLAV .1', 'PIP/TAZO.1', 'CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1', 'AZTREONAM.1', 'IMIPENEM.1']].loc[fen.index]


##################### TRAIN BASELINES AND PREDICT ######################
results = np.zeros((5,15))
phen = np.zeros((5,3))
feat_imp = np.zeros((5,3,10000))

for f in range(len(folds["train"])):
    print("Training fold: ", f)
    
    if "BTSE3" in folds["train"][f]: folds["train"][f].remove("BTSE3")
    if "BTSH11" in folds["train"][f]: folds["train"][f].remove("BTSH11")
    if "BTSE3" in folds["val"][f]: folds["train"][f].remove("BTSE3")
    if "BTSH11" in folds["val"][f]: folds["train"][f].remove("BTSH11")

    ab = ab.fillna(ab.mean())
    ab[ab>0.5] = 1
    ab[ab<0.5] = 0
    y_tr, y_val = ab.loc[folds["train"][f]], ab.loc[folds["val"][f]]
    # y_tr, y_val = ab.loc[folds["train"][f]].dropna(), ab.loc[folds["val"][f]]

    for idx in y_val.index:
        print(idx in y_tr.index)
    x0_tr, x0_val = maldi.loc[y_tr.index], maldi.loc[y_val.index]
    ###### PRED CARB BLEE
    ph_tr, ph_val = np.vstack(fen.loc[y_tr.index].values), np.vstack(fen.loc[y_val.index].values)

    x0_tr= np.vstack(x0_tr.values).astype(float)
    mean = np.mean(x0_tr)
    x0_tr /= mean
    x0_val = np.vstack(x0_val.values).astype(float)
    x0_val /= mean


    y_tr = y_tr.values
    y_val = y_val.values
    # ###### PRED CARB BLEE
    y_tr = np.hstack((ph_tr, y_tr))
    y_val = np.hstack((ph_val, y_val))

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
        kernel= DotProduct()
        CV_rfc = GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
        # CV_rfc = GaussianProcessClassifier(n_jobs=-1)

    if randomforest:
        for c in range(y_tr.shape[1]):
            CV_rfc.fit(x0_tr, y_tr[:, c])
            y_pred = CV_rfc.predict_proba(x0_val)[:, 1]
            score= auc(y_val[:, c], y_pred)
            results[f,c]=score
            if c<3:
                phen[f, c] = score
                feat_imp[f, c, :] = CV_rfc.best_estimator_.feature_importances_
    
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
            if c<3:
                phen[f, c] = score
                # feat_imp[f, c, :] = CV_rfc.best_estimator_.coef_
    else:
        for c in range(y_tr.shape[1]):
            CV_rfc.fit(x0_tr, y_tr[:, c])
            y_pred = CV_rfc.predict_proba(x0_val)
            y_pred = y_pred[:,1]
            score= auc(y_val[:, c], y_pred)
            if c<3:
                phen[f, c] = score
            else:
                results[f,c-3] = score


    print(results)

print("Results AB")
print(np.mean(results, axis=0))
print(np.std(results, axis=0))
print("Results RM")
print(np.mean(phen, axis=0))
print(np.std(phen, axis=0))


feat_imp_mean = np.mean(feat_imp, axis=0)
plt.figure(figsize=[15,10])
plt.plot(range(2000,2500), feat_imp_mean[0, 0:500], label="CARB+ESBL")
plt.plot(range(2000,2500), feat_imp_mean[1, 0:500], label="Only ESBL")
plt.plot(range(2000,2500), feat_imp_mean[2, 0:500], label="Susceptible")
plt.legend()

feat_imp_mean = np.mean(feat_imp, axis=0)
plt.figure(figsize=[15,10])
plt.plot(range(7000,7500), feat_imp_mean[0, 5000:5500], label="CARB+ESBL")
plt.plot(range(7000,7500), feat_imp_mean[1, 5000:5500], label="Only ESBL")
plt.plot(range(7000,7500), feat_imp_mean[2, 5000:5500], label="Susceptible")
plt.legend()

feat_imp_mean = np.mean(feat_imp, axis=0)
plt.figure(figsize=[15,10])
plt.plot(range(9500,10000), feat_imp_mean[0, 7500:8000], label="CARB+ESBL")
plt.plot(range(9500,10000), feat_imp_mean[1, 7500:8000], label="Only ESBL")
plt.plot(range(9500,10000), feat_imp_mean[2, 7500:8000], label="Susceptible")
plt.legend()