import numpy as np
import pickle
import sys
from sklearn.model_selection import KFold
sys.path.insert(0, "./ksshiba")
import preprocess_data as pp
from ksshiba import fast_fs_ksshiba_b as ksshiba

# Load only RM data
gm_path = "./data/old_data/"

# with open("./data/gm_data.pkl", 'rb') as pkl:
#     gm_data = pickle.load(pkl)
#
# gm_x, gm_y = gm_data[0], gm_data[1]
# del gm_data

gm_x, gm_y, gm_labels = pp.process_hospital_data(gm_path=gm_path, gm_n_labels=18, ryc_path='data/Klebsiellas_RyC/',
                                      keep_common=True, return_GM=True, return_RYC=False, norm_both=False)

y_seen = gm_y[~np.isnan(gm_y).any(axis=1)]
y_ss = gm_y[np.isnan(gm_y).any(axis=1)]
ss_index = np.argwhere(np.isnan(gm_y).any(axis=1))[:, 0]
x_seen = np.delete(gm_x, ss_index, axis=0)
x_ss = gm_x[ss_index]

# create_folds = False
#
# if create_folds:
#     kf = KFold(n_splits=10, random_state=32, shuffle=True)
#     gm_folds = {"train": [], "test": []}
#     for train_idx, test_idx in kf.split(x_seen):
#         gm_folds["train"].append(train_idx)
#         gm_folds["test"].append(test_idx)
#     with open("data/gm_folds.pkl", 'wb') as f:
#         pickle.dump(gm_folds, f)
#
#     del gm_folds

results = {}

with open("./data/gm_folds.pkl", 'rb') as pkl:
    folds = pickle.load(pkl)

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1e5)}}
c = 0
for f in range(len(folds["train"])):
    print("Training fold: ", c)
    x_tr, y_tr, x_tst, y_tst = x_seen[folds["train"][f]], y_seen[folds["train"][f]], x_seen[folds["test"][f]], y_seen[
        folds["test"][f]]

    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    # Concatenate the X fold of seen points and the unseen points
    X = np.vstack((x_ss, x_tr, x_tst))

    Y = np.vstack((y_ss, y_tr))

    X0_kernel_lin = myModel_mul.struct_data(X, method='reg', V=X, kernel="linear", sparse_fs=1)

    Y0_tr = myModel_mul.struct_data(Y, 'mult', 0)

    myModel_mul.fit(X0_kernel_lin, Y0_tr, max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1, ACC=1, AUC=1, feat_crit=1e-2)

    model_name = "model_fold" + str(c)
    results[model_name] = myModel_mul
    c += 1

with open("Results/GM_commonaxis_commonlabels_mult_model.pkl", 'wb') as f:
    pickle.dump(results, f)
