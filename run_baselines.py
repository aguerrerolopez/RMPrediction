import preprocess as pp
from train import train
from models import LR_CV

# Load data
data_path = "./data/old_data/"
data = pp.load_data(path=data_path, format="zip")

# Preprocess data
x, x_axis, y, labels_name = pp.prepare_data(data, drop=True, columns_drop=["AMPICILINA"],
                                            impute=True, limits=[2005,19805], n_labels=18)


# =================== LR with L1, L2 and Elasticnet ===========
# Folds creation
cv_folds = 5
# Threshold to decide predictions with predict_proba
th = 0.5

# LR with L1
lr_l1 = LR_CV(penalty='l1', solver='saga', class_weight='balanced',
              random_state=32, max_iter=10, refit=True)
# LR with L2
lr_l2 = LR_CV(penalty='l1', solver='saga', class_weight='balanced',
              random_state=32, max_iter=10, refit=True)
# LR with elasticnet
lr_enet = LR_CV(penalty='elasticnet', solver='saga', class_weight='balanced',
              random_state=32, max_iter=10, refit=True, l1_ratios=[0.3, 0.5, 0.8])


train(model=lr_l1, model_name="LR_L1", x=x, y=y, labels_list=labels_name.tolist(), folds=cv_folds, th=th)
train(model=lr_l2, model_name="LR_L2", x=x, y=y, labels_list=labels_name.tolist(), folds=cv_folds, th=th)
train(model=lr_enet, model_name="LR_enet", x=x, y=y, labels_list=labels_name.tolist(), folds=cv_folds, th=th)



