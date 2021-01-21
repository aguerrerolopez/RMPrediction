import numpy as np
import pickle
import sys
from sklearn.model_selection import KFold
sys.path.insert(0, "./ksshiba")
from ksshiba import fast_fs_ksshiba_b as ksshiba

# Load only RM data
# gm_path = "./data/old_data/"

with open("./data/ryc_data.pkl", 'rb') as pkl:
    ryc_full_data = pickle.load(pkl)

# ryc_full_data = pp.process_hospital_data(gm_path=gm_path, gm_n_labels=18, ryc_path='data/Klebsiellas_RyC/',
#                                       keep_common=True, return_GM=False, return_RYC=True, norm_both=False)

ryc_x = ryc_full_data["maldi"].copy(deep=True)
ryc_y = ryc_full_data[ryc_full_data.columns[-12:]].copy(deep=True)
nan_index = ryc_full_data.isnull().any(axis=1)

y_seen = ryc_y[~nan_index]
y_ss = ryc_y[nan_index]
x_seen = ryc_x[~nan_index]
x_ss = ryc_x[nan_index]

# create_ryc_folds = True
#
# if create_ryc_folds:
#     n_samples = np.unique(x_seen.index.values)
#     kf = KFold(n_splits=10, random_state=32, shuffle=True)
#
#     folds = {"train": [], "test": []}
#     for train_idx, test_idx in kf.split(range(len(n_samples))):
#       folds["train"].append(n_samples[train_idx])
#       folds["test"].append(n_samples[test_idx])
#
#     with open("data/ryc_folds.pkl", 'wb') as f:
#        pickle.dump(folds, f)

results = {}

with open("./data/ryc_folds.pkl", 'rb') as pkl:
    folds = pickle.load(pkl)

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1e5)}}
c = 0
for f in range(len(folds["train"])):
    print("Training fold: ", c)
    x_tr, y_tr, x_tst, y_tst = x_seen.loc[folds["train"][f]], y_seen.loc[folds["train"][f]],\
                               x_seen.loc[folds["test"][f]], y_seen.loc[folds["test"][f]]

    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    # Concatenate the X fold of seen points and the unseen points
    X = np.vstack((np.vstack(x_ss.values), np.vstack(x_tr.values), np.vstack(x_tst.values)))
    Y = np.vstack((np.vstack(y_ss.values), np.vstack(y_tr.values)))
    Y[Y > 1] = np.NaN

    X0_kernel_lin = myModel_mul.struct_data(X, method='reg', V=X, kernel="linear", sparse_fs=1)
    Y0_tr = myModel_mul.struct_data(Y, 'mult', 0)

    myModel_mul.fit(X0_kernel_lin, Y0_tr, max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1, ACC=1, AUC=1, feat_crit=1e-2)

    model_name = "model_fold" + str(c)
    results[model_name] = myModel_mul
    c += 1

with open("Results/RyC_commonaxis_commonlabels_mult_model.pkl", 'wb') as f:
    pickle.dump(results, f)
