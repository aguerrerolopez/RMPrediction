import numpy as np
# import preprocess_data as pp
import pickle
import sys
from sklearn.model_selection import KFold

sys.path.insert(0, "./ksshiba")
from ksshiba import fast_fs_ksshiba_b as ksshiba

# Load both data
gm_path = "./data/old_data/"
ryc_path = "./data/Klebsiellas_RyC/"

with open("./data/ryc_data_both.pkl", 'rb') as pkl:
    ryc_full_data = pickle.load(pkl)
with open("./data/gm_data_both.pkl", 'rb') as pkl:
    gm_data = pickle.load(pkl)

# gm_data, ryc_full_data = pp.process_hospital_data(gm_path=gm_path, gm_n_labels=18, ryc_path=ryc_path,
#                                                   keep_common=True, return_GM=True, return_RYC=True, norm_both=True)

ryc_x = ryc_full_data["maldi"].copy(deep=True)
ryc_y = ryc_full_data[ryc_full_data.columns[-12:]].copy(deep=True)
nan_index = ryc_full_data.isnull().any(axis=1)

gm_x, gm_y = gm_data[0], gm_data[1]

gm_y_seen = gm_y[~np.isnan(gm_y).any(axis=1)]
gm_y_ss = gm_y[np.isnan(gm_y).any(axis=1)]
ss_index = np.argwhere(np.isnan(gm_y).any(axis=1))[:, 0]
gm_x_seen = np.delete(gm_x, ss_index, axis=0)
gm_x_ss = gm_x[ss_index]

ryc_y_seen = ryc_y[~nan_index]
ryc_y_ss = ryc_y[nan_index]
ryc_x_seen = ryc_x[~nan_index]
ryc_x_ss = ryc_x[nan_index]

results = {}

with open("./data/ryc_folds.pkl", 'rb') as pkl:
    ryc_folds = pickle.load(pkl)

with open("./data/gm_folds.pkl", 'rb') as pkl:
    gm_folds = pickle.load(pkl)

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1e5)}}

for f in range(len(gm_folds["train"])):
    print("Training fold: ", f)

    # ============= LOAD FOLDS ==================
    ryc_x_tr, ryc_y_tr, ryc_x_tst, ryc_y_tst = ryc_x_seen.loc[ryc_folds["train"][f]], \
                                               ryc_y_seen.loc[ryc_folds["train"][f]], \
                                               ryc_x_seen.loc[ryc_folds["test"][f]], \
                                               ryc_y_seen.loc[ryc_folds["test"][f]]
    gm_x_tr, gm_y_tr, gm_x_tst, gm_y_tst = gm_x_seen[gm_folds["train"][f]], gm_y_seen[gm_folds["train"][f]], \
                                           gm_x_seen[gm_folds["test"][f]], gm_y_seen[gm_folds["test"][f]]

    # ============= CONCATENATE BOTH HOSPITAL DATA ==================
    ryc_Y = np.vstack((np.vstack(ryc_y_ss.values), np.vstack(ryc_y_tr.values)))

    X = np.vstack((np.vstack(ryc_x_ss.values), gm_x_ss,
                   np.vstack(ryc_x_tr.values), gm_x_tr,
                   np.vstack(ryc_x_tst.values), gm_x_tst))
    Y = np.vstack((np.vstack(ryc_y_ss.values), gm_y_ss,
                   np.vstack(ryc_y_tr.values), gm_y_tr))
    Y[Y > 1] = np.NaN

    # ============= INITIALIZE MODEL PARAMETERS ==================
    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    X0_kernel_lin = myModel_mul.struct_data(X, method='reg', V=X, kernel="linear", sparse_fs=1)
    Y0_tr = myModel_mul.struct_data(Y, 'mult', 0)

    myModel_mul.fit(X0_kernel_lin, Y0_tr, max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1, ACC=1, AUC=1, feat_crit=1e-2)

    model_name = "model_fold" + str(f)
    results[model_name] = myModel_mul

with open("Results/RyC-GM_mult_model.pkl", 'wb') as f:
    pickle.dump(results, f)
