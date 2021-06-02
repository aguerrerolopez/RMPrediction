import pickle
import numpy as np
import json
from numpy.testing._private.utils import KnownFailureException
import telegram
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)


# FAMILIAS
familias = { "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
                    "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
                    "monobactams": ['AZTREONAM'],
                    "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
                    "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA'],
                    "fluoro":['CIPROFLOXACINO'],
                    "otros":['COLISTINA']
                    }

full = True

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
        hgm_data = pickle.load(pkl)

random_states = [0, 10, 20, 30, 40]
results = np.zeros((10,15))

feat_imp_byfam = np.zeros((7, 10000))
feat_imp_byfam_std =  np.zeros((7, 10000))
for i, familia in enumerate(familias):
    if familia=="cephalos":
        delay=len(familias["penicilinas"])
    elif familia=="monobactams":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])
    elif familia=="carbapenems":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])+len(familias["monobactams"])
    elif familia=="aminos":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])+len(familias["monobactams"])+len(familias["carbapenems"])
    elif familia=="fluoro":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])+len(familias["monobactams"])+len(familias["carbapenems"])+len(familias["aminos"])
    elif familia=="otros":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])+len(familias["monobactams"])+len(familias["carbapenems"])+len(familias["aminos"])+len(familias["fluoro"])
    else:
        delay=0
    
    folds_path = "./data/HGM_10STRATIFIEDfolds_nomissing_muestrascompensadas_"+familia+".pkl"
    with open(folds_path, 'rb') as pkl:
        hgm_folds = pickle.load(pkl)
    # plt.figure()

    feat_imp_byfold = np.zeros((10, 10000))
    for fold in range(10):
        x_tr = np.vstack(hgm_data['maldi'].loc[hgm_folds["train"][fold]].values)
        x_tst = np.vstack(hgm_data['maldi'].loc[hgm_folds["val"][fold]].values)

        y_tr = hgm_data['binary_ab'][familias[familia]].loc[hgm_folds["train"][fold]].values
        y_tst = hgm_data['binary_ab'][familias[familia]].loc[hgm_folds["val"][fold]].values
        
        # if y_tst.shape[1]==1:
        #     y_tr = y_tr.ravel()
        #     clf=SVC(probability=1, kernel="linear")
        #     param_grid = {'C': [0.01, 0.1 , 1, 10]}
        # else:
        #     clf = MultiOutputClassifier(SVC(probability=True, kernel="linear"))
        #     param_grid = {'estimator__C': [0.01, 0.1 , 1, 10]}     
        #     # knn = KNeighborsClassifier(n_jobs=-1)
        #     # param_grid = {'n_neighbors': range(1,20)}
        #     # rfc=RFC(random_state=42)
        #     # param_grid = {'n_estimators': [100],
        #                     # 'max_features': ['auto', 'sqrt', 'log2'],
        #                     # 'criterion' :['gini']}
        clf = RFC(n_jobs=-1)
        param_grid = {'n_estimators': [100],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'criterion' :['gini']}

        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        CV_rfc.fit(x_tr, y_tr)

        print(CV_rfc.best_params_)
        y_pred_clf = CV_rfc.predict_proba(x_tst)
        y_pred = np.zeros(y_tst.shape)
        if y_tst.shape[1]==1:
            y_pred = y_pred_clf[:, 1]
            results[fold, delay] = auc(y_tst, y_pred)
        else:    
            for c in range(len(y_pred_clf)):
                if familia=="carbapenems":
                    print(y_tst[:, c])
                    print(y_pred_clf[c][:, 1])
                results[fold, c+delay] = auc(y_tst[:, c], y_pred_clf[c][:, 1])

        feat_imp_byfold[fold, :] = CV_rfc.best_estimator_.feature_importances_.ravel()

    feat_imp_byfam[i, :] = np.mean(feat_imp_byfold, axis=0)
    feat_imp_byfam_std[i,:] =np.std(feat_imp_byfold, axis=0)
    #     plt.plot(CV_rfc.best_estimator_.feature_importances_, label="| Fold:" +str(fold))
    # plt.legend()
    # plt.title("Familia: "+familia)
    # plt.show()

print(results)

for i, fam in enumerate(familias):
    plt.figure(figsize=[20, 10])
    plt.title("Feature importance in mean and std in "+fam)
    plt.errorbar(x=range(0,10000), y=feat_imp_byfam[i,:], yerr=feat_imp_byfam_std[i,:], fmt='o', color='black',
             ecolor='lightgray',  alpha=0.3, label=fam)
    plt.show()


# JUST A SAMPLE
familia="carbapenems"
for ab in familias[familia]:
    plt.figure(figsize=[20, 10])
    plt.title("Resistant and sensible sample for "+ab)
    plt.plot(feat_imp_byfam[3, :],marker='o', color="black", alpha=0.2, label="W primal space MEAN of "+familia)
    pos_sample = np.vstack(hgm_data['maldi'].loc[hgm_data['binary_ab'][ab][hgm_data['binary_ab'][ab]==1].sample(1).index]).ravel()
    neg_sample = np.vstack(hgm_data['maldi'].loc[hgm_data['binary_ab'][ab][hgm_data['binary_ab'][ab]==0].sample(1).index]).ravel()
    plt.plot(pos_sample, color='green', label=ab+": Resistent sample")
    plt.plot(neg_sample, color='orange', label=ab+": Sensible sample")
    plt.legend()
    plt.show()

# MEAN OF THE POS AND NEG SAMPLE

for ab in familias[familia]:
    plt.figure(figsize=[20, 10])
    plt.title("Resistant and sensible mean for "+ab)
    plt.plot(feat_imp_byfam[3, :],marker='o', color="black", alpha=0.2, label="W primal space MEAN of "+familia)
    pos_sample = np.mean(np.vstack(hgm_data['maldi'].loc[hgm_data['binary_ab'][ab][hgm_data['binary_ab'][ab]==1].index]), axis=0)
    neg_sample = np.mean(np.vstack(hgm_data['maldi'].loc[hgm_data['binary_ab'][ab][hgm_data['binary_ab'][ab]==0].index]), axis=0)
    plt.plot(pos_sample, color='green', label=ab+": Resistent MEAN")
    plt.plot(neg_sample, color='orange', label=ab+": Sensible MEAN")
    plt.legend()
    plt.show()






