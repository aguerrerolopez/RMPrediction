import pickle
import sys
import numpy as np
sys.path.insert(0, "./lib")
import json
import telegram
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score as auc
import pandas as pd

##################### PARAMETERS SELECTION ####################3
# Which baseline do you want to use:
svm = 0
knn = 0
rf = 1
gp = 0


########################## LOAD DATA ######################
data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    hgm_data = pickle.load(pkl)

folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_pruebaloca.pkl"
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)

data_path = "./data/ryc_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    ryc_data = pickle.load(pkl)

# COLUMNS TO USE FROM HGM
ab_cols =  ['AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM']

old_fen = hgm_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
hgm_data['fen'] = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

hgm_data['maldi'] = hgm_data['maldi'].loc[hgm_data['fen'].index]
hgm_data['cmi'] = hgm_data['cmi'].loc[hgm_data['fen'].index]
hgm_data['binary_ab'] = hgm_data['binary_ab'][ab_cols].loc[hgm_data['fen'].index]

# COLUMNS TO USE FROM HRC
ab_cols =  ['AMOXI/CLAV .1', 'PIP/TAZO.1', 'CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1', 'AZTREONAM.1', 'IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']

old_fen = ryc_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
ryc_data['fen'] = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

full_predict = pd.concat([ryc_data['fen'], ryc_data['binary_ab'][ab_cols].loc[ryc_data['fen'].index]], axis=1)
# REMOVE MISSING DATA FOR HRC DATA
full_predict = full_predict.dropna()

ryc_data['maldi'] = ryc_data['maldi'].loc[full_predict.index]
ryc_data['binary_ab'] = full_predict[ab_cols]
ryc_data['fen'] = full_predict[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

# REMOVE MISSING DATA FOR HGM DATA
nomissing = hgm_data['binary_ab'].loc[folds["train"][0]].dropna().index

############### CONSTRUCT TRAIN AND TEST DATA
# MALDI MS DATA FROM HGM
hgm_x0_tr, hgm_x0_tst = np.vstack(hgm_data['maldi'].loc[nomissing].values), np.vstack(hgm_data['maldi'].loc[folds["val"][0]].values)

# RESISTANCE MECHANISM DATA FROM HGM
hgm_x1_tr, hgm_x1_tst = hgm_data['fen'].loc[nomissing], hgm_data['fen'].loc[folds["val"][0]]
# AMR DATA FROM HGM
hgm_y_tr, hgm_y_tst = hgm_data['binary_ab'].loc[nomissing], hgm_data['binary_ab'].loc[folds["val"][0]]

# X_TRAIN = MALDI MS DATA HGM
x0_tr = np.vstack((hgm_x0_tr, hgm_x0_tst))
# X_TEST = MALDI MS DATA HRC
x0_tst = np.vstack(ryc_data['maldi'].values)

# Y_TRAIN = RESISTANCE MECHANISM + AMR DATA FROM HGM
x1 = np.vstack((hgm_x1_tr.values, hgm_x1_tst.values))
y0 = np.vstack((hgm_y_tr.values, hgm_y_tst.values))
y_tr = np.hstack((x1,y0))
# Y_TEST = RESISTANCE MECHANISM + AMR DATA FROM HRC
y_tst = np.hstack((ryc_data['fen'], ryc_data['binary_ab']))


##################### TRAIN BASELINES AND PREDICT ######################

if rf:
    from sklearn.ensemble import RandomForestClassifier as RFC
    clf = RFC(n_jobs=-1)
    param_grid = {'n_estimators': [50, 100, 200],
                  'max_features': ['auto', 'sqrt', 'log2']}
if knn:
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_jobs=-1)
    param_grid = {'n_neighbors': range(1,20)}
if svm:
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.svm import SVC
    clf = MultiOutputClassifier(SVC(probability=True, kernel="linear"))
    param_grid = {'estimator__C': [0.01, 0.1 , 1, 10]}  
if gp:
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import DotProduct
    kernel= DotProduct()
    CV_rfc = MultiOutputClassifier(GaussianProcessClassifier(kernel=kernel, n_jobs=-1))

results = np.zeros((1,y_tr.shape[1]))
if gp:
    CV_rfc.fit(x0_tr, y_tr)
    y_pred = CV_rfc.predict_proba(x0_tst)

    print(y_tst.shape)
    print(y_pred[0].shape)
    print(y_tr.shape)
    for c in range(y_tr.shape[1]):
        results[0, c] = auc(y_tst[:, c], y_pred[c][:,1])
else:
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    CV_rfc.fit(x0_tr, y_tr)
    y_pred = CV_rfc.predict_proba(x0_tst)
    print(y_tst.shape)
    print(y_pred[0].shape)
    print(y_tr.shape)
    for c in range(y_tr.shape[1]):
        results[0, c] = auc(y_tst[:, c], y_pred[c][:,1])
    


print(results)
print(np.mean(results, axis=0))
print(np.std(results, axis=0))