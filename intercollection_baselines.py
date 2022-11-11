#%%
import pickle
import sys
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
import numpy as np
sys.path.insert(0, "./lib")
import json
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score as auc
import lightgbm as lgb



################## LOAD DATA ###################3

data_path = "./data/ryc_data_processed.pkl"
with open(data_path, 'rb') as pkl:
    ramon_data = pickle.load(pkl)
data_path = "./data/gm_data_processed.pkl"
with open(data_path, 'rb') as pkl:
    greg_data = pickle.load(pkl)

fen_greg = greg_data['full'][['CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
maldi_greg = greg_data['maldi'].loc[fen_greg.index]
fen_ramon = ramon_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
maldi_ramon = ramon_data['maldi'].loc[fen_ramon.index]

###### LOAD FOLDS
folds_path = "./data/GM_5STRATIFIEDfolds_paper.pkl"
with open(folds_path, 'rb') as pkl:
    folds_greg = pickle.load(pkl)
folds_path = "./data/RYC_5STRATIFIEDfolds_paper.pkl"
with open(folds_path, 'rb') as pkl:
    folds_ramon = pickle.load(pkl) 

#%%
################################# EXPERIMENT 2: PREDICT OXA48 in both hospitals without hospital indicator label########################################33

##################### PARAMETERS SELECTION ####################
# KERNEL SELECTION: linear or rbf
kernel = "rbf"
# MODEL SELECTION: put a 1 into the model selected and 0s in all the toher options
randomforest = 0
knn=0
svm=0
gp=0
mlp=0
lr = 0
xgboost_flag = 1
lgb_flag = 0

######## BUILD THE MODEL
if randomforest:
    from sklearn.ensemble import RandomForestClassifier as RFC
    clf = RFC(n_jobs=-1)
    # RF CV parameters
    param_grid = {'n_estimators': [50, 100, 150],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'criterion' :['gini']}
    # Intra 5 CV
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
elif lgb_flag:
    lgbparams = {
                'learning_rate': [0.005, 0.01],
                'n_estimators': [8,16,24],
                'num_leaves': [6,8,12,16], # large num_leaves helps improve accuracy but might lead to over-fitting
                'boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
                'objective' : ['binary'],
                }
    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt', 
                    objective = 'binary', 
                    n_jobs = -1)
    CV_rfc = GridSearchCV(mdl, lgbparams, verbose=2, cv=5, n_jobs=-1)
elif mlp:
    from sklearn.neural_network import MLPClassifier
    parameter_space = {
    'hidden_layer_sizes': [(500,100,10),(500,250,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}
    mlp_gs = MLPClassifier(max_iter=100)
    CV_rfc = GridSearchCV(estimator=mlp_gs, param_grid=parameter_space, cv=5, n_jobs=-1, verbose=1)
elif xgboost_flag:
    from xgboost import XGBClassifier
    params = {
        'min_child_weight': [1, 5],
        'gamma': [0.5, 1, 1.5],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'max_depth': [3, 4, 5]
        }
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic')
    CV_rfc = GridSearchCV(estimator=xgb, param_grid=params, cv=5, verbose=1, n_jobs=-1)
elif knn:
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_jobs=-1)
    # RF CV parameters
    param_grid = {'n_neighbors': range(1,20)}
    # Intra 5 CV
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
elif svm:
    from sklearn.svm import SVC
    clf = SVC(probability=True, kernel=kernel)
    # RF CV parameters
    param_grid = {'C': [0.01, 0.1 , 1, 10]}
    # Intra 5 CV
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
elif gp:
    # No CV parameters because is a GP, do not need CV.
    from sklearn.gaussian_process import GaussianProcessClassifier
    if kernel=="linear":
        from sklearn.gaussian_process.kernels import DotProduct
        kernel= DotProduct()
        CV_rfc = GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
    if kernel=="rbf":
        CV_rfc = GaussianProcessClassifier(n_jobs=-1)


############ TRAIN THE MODEL SELECTED
greg = [[], [], []]
ramon = [[], [], []]
for r in range(5):
    print("Training model "+str(r))
    ###### PREPARE DATA GREGORIO
    ## Maldi DATA
    x0_tr, x0_val = maldi_greg.loc[folds_greg["train"][r]], maldi_greg.loc[folds_greg["val"][r]]
    x0_greg_train= np.vstack(x0_tr.values).astype(float)
    x0_greg_test = np.vstack(x0_val.values).astype(float)
    x0_greg_test /= np.mean(x0_greg_train)
    x0_greg_train /= np.mean(x0_greg_train)

    # AST
    x1_tr, x1_val = fen_greg.loc[folds_greg["train"][r]], fen_greg.loc[folds_greg["val"][r]]
    x1_greg_train = np.vstack(x1_tr.values)
    x1_greg_test = np.vstack(x1_val.values)

    ###### PREPARE DATA RAMON
    # AST
    x1_tr, x1_val = fen_ramon.loc[folds_ramon["train"][r]].dropna(), fen_ramon.loc[folds_ramon["val"][r]].dropna()
    x1_ramon_train = np.vstack(x1_tr.values)
    x1_ramon_test = np.vstack(x1_val.values)

    ## Maldi DATA
    x0_tr, x0_val = maldi_ramon.loc[x1_tr.index], maldi_ramon.loc[x1_val.index]
    x0_ramon_train= np.vstack(x0_tr.values).astype(float)
    x0_ramon_test = np.vstack(x0_val.values).astype(float)
    x0_ramon_test /= np.mean(x0_ramon_train)
    x0_ramon_train /= np.mean(x0_ramon_train)

    # CONCATENATE BOTH COLLECTION DATA INTO ONE MATRIX
    x_train = np.vstack((x0_greg_train, x0_ramon_train)) 
    y_train = np.vstack((x1_greg_train, x1_ramon_train)) 

    # FIT AND PREDICT THE MODEL FOR EACH PREDICTION TASK
    for c in range(x1_ramon_train.shape[1]):
    
        CV_rfc.fit(x_train, y_train[:, c])
        y_pred = CV_rfc.predict_proba(x0_greg_test)[:, 1]
        y_pred_ramon = CV_rfc.predict_proba(x0_ramon_test)[:, 1]

        greg[c].append(auc(x1_greg_test[:, c], y_pred))
        ramon[c].append(auc(x1_ramon_test[:, c], y_pred_ramon))



print("######### RESULTS")

print("AUC in GM collection")
print("ESBL+CP: "+str(np.mean(greg[0]))+"+-"+str(np.std(greg[0])))
print("ESBL: "+str(np.mean(greg[1]))+"+-"+str(np.std(greg[1])))
print("WT: "+str(np.mean(greg[2]))+"+-"+str(np.std(greg[2])))

print("AUC in RyC collection")
print("ESBL+CP: "+str(np.mean(ramon[0]))+"+-"+str(np.std(ramon[0])))
print("ESBL: "+str(np.mean(ramon[1]))+"+-"+str(np.std(ramon[1])))
print("WT: "+str(np.mean(ramon[2]))+"+-"+str(np.std(ramon[2])))

