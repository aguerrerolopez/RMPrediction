import pickle
import sys
import numpy as np
sys.path.insert(0, "./lib")
from lib import fast_fs_ksshiba_b_ord as ksshiba
import json
import telegram
import topf
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
from data import MaldiTofSpectrum
from kernels import DiffusionKernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score as auc

def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)


hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_noCP.pkl"
# folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_noCP.pkl"
# folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_resto.pkl"
data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"


with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)

###### PRED CARB BLEE
old_fen = gm_data['fen']
# old_fen['Fenotipo CP+ESBL'][old_fen['Fenotipo CP']==1] = 1
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
fen = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

maldi = gm_data['maldi'].loc[fen.index]
cmi = gm_data['cmi'].loc[fen.index]
ab = gm_data['binary_ab'][['AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM']].loc[fen.index]

###### PRED RESTO
# maldi = gm_data['maldi']
# cmi = gm_data['cmi']
# ab = gm_data['binary_ab'][['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA', 'CIPROFLOXACINO', 'COLISTINA']]

results = np.zeros((10,15))
phen = np.zeros((10,3))

with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)


randomforest = 0
knn=0
svm=1
gp=0

feat_imp = np.zeros((10,3,10000))
for f in range(len(folds["train"])):
    print("Training fold: ", f)

    y_tr, y_val = ab.loc[folds["train"][f]].dropna(), ab.loc[folds["val"][f]] 
    for idx in y_val.index:
        print(idx in y_tr.index)
    x0_tr, x0_val = maldi.loc[y_tr.index], maldi.loc[y_val.index]
    ###### PRED CARB BLEE
    ph_tr, ph_val = np.vstack(fen.loc[y_tr.index].values), np.vstack(fen.loc[y_val.index].values)

    x0_tr= np.vstack(x0_tr.values).astype(float)
    x0_val = np.vstack(x0_val.values).astype(float)
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
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.svm import SVC
        # clf = MultiOutputClassifier(SVC(probability=True, kernel="rbf"))
        clf = SVC(probability=True, kernel="rbf")
        # param_grid = {'estimator__C': [0.01, 0.1 , 1, 10]}
        param_grid = {'C': [0.01, 0.1 , 1, 10]}
        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    elif gp:
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import DotProduct
        # kernel= DotProduct()
        # CV_rfc = MultiOutputClassifier(GaussianProcessClassifier(kernel=kernel, n_jobs=-1))
        CV_rfc = MultiOutputClassifier(GaussianProcessClassifier(n_jobs=-1))

    if randomforest:
        for c in range(y_tr.shape[1]):
            CV_rfc.fit(x0_tr, y_tr[:, c])
            y_pred = CV_rfc.predict_proba(x0_val)[:, 1]
            score= auc(y_val[:, c], y_pred)
            results[f,c]=score
            # if c<3:
            #     phen[f, c] = score
            #     feat_imp[f, c, :] = CV_rfc.best_estimator_.feature_importances_
            # else:
            #     results[f,c-3] = score
            
    else:
        
        for c in range(y_tr.shape[1]):
            if c>0:
                x0_tr = np.hstack((x0_tr, y_tr[:, c-1][:, np.newaxis]))
                x0_val = np.hstack((x0_val, y_pred[:, np.newaxis]))
            CV_rfc.fit(x0_tr, y_tr[:, c])
            y_pred = CV_rfc.predict_proba(x0_val)
            y_pred = y_pred[:,1]
            score= auc(y_val[:, c], y_pred)
            # ph_pred = y_pred[c][:,1]
            # score= auc(y_val[:, c], ph_pred)
            # results[f,c]=score
            if c<3:
                phen[f, c] = score
            else:
                results[f,c-3] = score
    print(results)

print("RESULTADOS 15 AB")
print(results)
print(np.mean(results, axis=0))
print(np.std(results, axis=0))
print("RESULTADOS PHENOTIPO")
print(phen)
print(np.mean(phen, axis=0))
print(np.std(phen, axis=0))


# # for i, fam in enumerate(familias):
#     plt.figure(figsize=[20, 10])
#     plt.title("Feature importance in mean and std in "+fam)
#     plt.errorbar(x=range(0,10000), y=feat_imp_byfam[i,:], yerr=feat_imp_byfam_std[i,:], fmt='o', color='black',
#              ecolor='lightgray',  alpha=0.3, label=fam)
#     plt.show()


# # JUST A SAMPLE
# familia="carbapenems"
# for ab in familias[familia]:
#     plt.figure(figsize=[20, 10])
#     plt.title("Resistant and sensible sample for "+ab)
#     plt.plot(feat_imp_byfam[3, :],marker='o', color="black", alpha=0.2, label="W primal space MEAN of "+familia)
#     pos_sample = np.vstack(hgm_data['maldi'].loc[hgm_data['binary_ab'][ab][hgm_data['binary_ab'][ab]==1].sample(1).index]).ravel()
#     neg_sample = np.vstack(hgm_data['maldi'].loc[hgm_data['binary_ab'][ab][hgm_data['binary_ab'][ab]==0].sample(1).index]).ravel()
#     plt.plot(pos_sample, color='green', label=ab+": Resistent sample")
#     plt.plot(neg_sample, color='orange', label=ab+": Sensible sample")
#     plt.legend()
#     plt.show()

# # MEAN OF THE POS AND NEG SAMPLE

# for ab in familias[familia]:
#     plt.figure(figsize=[20, 10])
#     plt.title("Resistant and sensible mean for "+ab)
#     plt.plot(feat_imp_byfam[3, :],marker='o', color="black", alpha=0.2, label="W primal space MEAN of "+familia)
#     pos_sample = np.mean(np.vstack(hgm_data['maldi'].loc[hgm_data['binary_ab'][ab][hgm_data['binary_ab'][ab]==1].index]), axis=0)
#     neg_sample = np.mean(np.vstack(hgm_data['maldi'].loc[hgm_data['binary_ab'][ab][hgm_data['binary_ab'][ab]==0].index]), axis=0)
#     plt.plot(pos_sample, color='green', label=ab+": Resistent MEAN")
#     plt.plot(neg_sample, color='orange', label=ab+": Sensible MEAN")
#     plt.legend()
#     plt.show()






