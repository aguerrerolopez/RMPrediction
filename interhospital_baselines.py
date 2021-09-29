#%%
import pickle
import sys
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
import numpy as np
sys.path.insert(0, "./lib")
import json
import telegram

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score as auc

# Telegram bot
def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)


################## LOAD DATA ###################3

data_path = "./data/ryc_data_expAug.pkl"
with open(data_path, 'rb') as pkl:
    ramon_data = pickle.load(pkl)
data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    greg_data = pickle.load(pkl)

fen_greg = greg_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
maldi_greg = greg_data['maldi'].loc[fen_greg.index]

fen_ramon = ramon_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
maldi_ramon = ramon_data['maldi'].loc[fen_ramon.index]

#%%
################################# EXPERIMENT 2: PREDICT OXA48 in both hospitals without hospital indicator label########################################33

###### PREPARE MODEL

kernel = "rbf"
randomforest = 0
knn=0
svm=1
gp=0

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
    clf = SVC(probability=True, kernel=kernel)
    param_grid = {'C': [0.01, 0.1 , 1, 10]}
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
elif gp:
    from sklearn.gaussian_process import GaussianProcessClassifier
    if kernel=="linear":
        from sklearn.gaussian_process.kernels import DotProduct
        kernel= DotProduct()
        CV_rfc = GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
    if kernel=="rbf":
        CV_rfc = GaussianProcessClassifier(n_jobs=-1)

###### LOAD FOLDS
folds_path = "./data/HGM_5STRATIFIEDfolds_muestrascompensada_resto.pkl"
with open(folds_path, 'rb') as pkl:
    folds_greg = pickle.load(pkl)

folds_path = "./data/RYC_5STRATIFIEDfolds_muestrascompensada_experimentAug.pkl"
with open(folds_path, 'rb') as pkl:
    folds_ramon = pickle.load(pkl) 

greg = [[], [], []]
ramon = [[], [], []]
for r in range(5):
    print("Training model "+str(r))
    ###### PREPARE DATA GREGORIO
    ## Maldi DATA
    x0_tr, x0_val = maldi_greg.loc[folds_greg["train"][r]], maldi_greg.loc[folds_greg["val"][r]]

    # Prepare train and test data
    x0_greg_train= np.vstack(x0_tr.values).astype(float)
    x0_greg_test = np.vstack(x0_val.values).astype(float)

    x0_greg_test /= np.mean(x0_greg_train)
    x0_greg_train /= np.mean(x0_greg_train)
    # # Fenotype
    x1_tr, x1_val = fen_greg.loc[folds_greg["train"][r]], fen_greg.loc[folds_greg["val"][r]]

    x1_greg_train = np.vstack(x1_tr.values)
    x1_greg_test = np.vstack(x1_val.values)

    ###### PREPARE DATA RAMON
    x1_tr, x1_val = fen_ramon.loc[folds_ramon["train"][r]].dropna(), fen_ramon.loc[folds_ramon["val"][r]].dropna()
    x0_tr, x0_val = maldi_ramon.loc[x1_tr.index], maldi_ramon.loc[x1_val.index]

    # Prepare train and test data
    x0_ramon_train= np.vstack(x0_tr.values).astype(float)
    x0_ramon_test = np.vstack(x0_val.values).astype(float)

    x0_ramon_test /= np.mean(x0_ramon_train)
    x0_ramon_train /= np.mean(x0_ramon_train)
    # # Fenotype
    x1_ramon_train = np.vstack(x1_tr.values)
    x1_ramon_test = np.vstack(x1_val.values)

    
    x_train = np.vstack((x0_greg_train, x0_ramon_train)) 
    y_train = np.vstack((x1_greg_train, x1_ramon_train)) 

    for c in range(x1_ramon_train.shape[1]):
        CV_rfc.fit(x_train, y_train[:, c])

        y_pred = CV_rfc.predict_proba(x0_greg_test)[:, 1]
        greg[c].append(auc(x1_greg_test[:, c], y_pred))

        y_pred = CV_rfc.predict_proba(x0_ramon_test)[:, 1]
        ramon[c].append(auc(x1_ramon_test[:, c], y_pred))

    ##### PREPARE VIEWS TRAIN/TEST

print("######### RESULTS")

print("AUC in Greg hospital")
print("CARB+BLEE: "+str(np.mean(greg[0]))+"+-"+str(np.std(greg[0])))
print("BLEE: "+str(np.mean(greg[1]))+"+-"+str(np.std(greg[1])))
print("sus: "+str(np.mean(greg[2]))+"+-"+str(np.std(greg[2])))

print("AUC in HRC hospital")
print("CARB+BLEE: "+str(np.mean(ramon[0]))+"+-"+str(np.std(ramon[0])))
print("BLEE: "+str(np.mean(ramon[1]))+"+-"+str(np.std(ramon[1])))
print("sus: "+str(np.mean(ramon[2]))+"+-"+str(np.std(ramon[2])))
# %%
