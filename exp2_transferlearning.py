#%%
import pickle
import sys
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
import numpy as np
sys.path.insert(0, "./lib")
from lib import fast_fs_ksshiba_b_ord as ksshiba
import json
import telegram
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

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

#%%
################################# EXPERIMENT 1: PREDICT OXA48 TRAINING HGM AND RETRAINING HRC ########################################33


###### PREPARE DATA GREGORIO
greg_maldi = np.vstack(greg_data['maldi'].values)
greg_maldi /= np.mean(greg_maldi)
greg_target = (greg_data['full']['Fenotipo CP+ESBL'].values+greg_data['full']['Fenotipo CP'].values)[:, np.newaxis]

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 100, "max_it": int(5000)}}


auc_list = []
for r in range(5):
    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    maldi_view_greg = myModel_mul.struct_data(greg_maldi, method="reg", V=greg_maldi, kernel='linear')
    gen_view = myModel_mul.struct_data(greg_target, method="mult")
    myModel_mul.fit(maldi_view_greg,
                    gen_view,
                    max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1,
                    feat_crit=1e-2)


    ##### Store r.v. learned:
    W_init = myModel_mul.q_dist.W
    tau_init = myModel_mul.q_dist.tau
    b_init = myModel_mul.q_dist.b
    alpha_init = myModel_mul.q_dist.alpha
    gamma_init = myModel_mul.q_dist.gamma

    ######## Now retrain with RAMON data

    ###### PREPARE DATA RAMON
    ramon_maldi = np.vstack(ramon_data['maldi'].values)
    ramon_maldi /= np.mean(ramon_maldi)
    ramon_target = ramon_data['full']['CP gene'].values.astype(int)[:, np.newaxis]
    ramon_indicator = np.ones(ramon_target.shape)

    from sklearn.model_selection import StratifiedShuffleSplit
    ss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
    r_idx,_ = ss1.split(ramon_maldi, ramon_indicator)
    r_idx_train=r_idx[0]
    r_idx_test=r_idx[1]

    ##### PREPARE VIEWS TRAIN/TEST

    maldi_view = np.vstack((ramon_maldi[r_idx_train], ramon_maldi[r_idx_test]))
    gen_view = np.vstack((ramon_target[r_idx_train]))

    ###### PREPARE MODEL

    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1,
                                W_init=W_init, b_init=b_init, alpha_init=alpha_init, tau_init=tau_init, gamma_init=gamma_init)

    maldi_view = myModel_mul.struct_data(maldi_view, method="reg", V=greg_maldi, kernel='linear')
    gen_view = myModel_mul.struct_data(gen_view, method="mult")

    myModel_mul.fit(maldi_view,
                    gen_view,
                    max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1,
                    feat_crit=1e-2)

    print("################ Results EXPERIMENT 1: OXA48 TRAINING HGM AND RETRAINING HRC##################")

    n_ramon = ramon_target[r_idx_test].shape[0]
    print("AUC predicting OXA48 IN RAMÓN Y CAJAL HOSPITAL")
    ramon_prediction = myModel_mul.t[1]['mean'][-n_ramon:,:]
    auc_carb = roc_auc_score(ramon_target[r_idx_test], ramon_prediction)
    print(auc_carb)
    auc_list.append(auc_carb)

print(np.mean(auc_list))
print(np.std(auc_list))


#%%
################################# EXPERIMENT 2: TRANSFER LEARNING ########################################33


###### PREPARE DATA GREGORIO
greg_maldi = np.vstack(greg_data['maldi'].values)
greg_maldi /= np.mean(greg_maldi)
greg_target = (greg_data['full']['Fenotipo CP+ESBL'].values+greg_data['full']['Fenotipo CP'].values)[:, np.newaxis]

auc_list_1=[]
auc_list_2=[]
for r in range(5):
    hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 100, "max_it": int(5000)}}

    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    maldi_view_greg = myModel_mul.struct_data(greg_maldi, method="reg", V=greg_maldi, kernel='linear')
    gen_view = myModel_mul.struct_data(greg_target, method="mult")

    myModel_mul.fit(maldi_view_greg,
                    gen_view,
                    max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1,
                    feat_crit=1e-2)

    ######## Now predict RAMON data

    ###### PREPARE DATA RAMON
    ramon_maldi = np.vstack(ramon_data['maldi'].values)
    ramon_maldi /= np.mean(ramon_maldi)
    ramon_target = ramon_data['full']['CP gene'].values.astype(int)[:, np.newaxis]
    ramon_indicator = np.ones(ramon_target.shape)
    
    from sklearn.model_selection import StratifiedShuffleSplit
    ss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
    r_idx,_ = ss1.split(ramon_maldi, ramon_indicator)
    r_idx_train=r_idx[0]
    r_idx_test=r_idx[1]
    ##### PREPARE VIEWS TRAIN/TEST

    maldi_view_train = np.vstack((ramon_maldi[r_idx_train]))
    maldi_view_test = np.vstack((ramon_maldi[r_idx_test]))

    gen_view_train = np.vstack((ramon_target[r_idx_train]))
    gen_view_test = np.vstack((ramon_target[r_idx_test]))

    ###### PREDICT WITH TRAINED MODEL.

    maldi_view_train = myModel_mul.struct_data(maldi_view_train, method="reg", V=greg_maldi, kernel='linear')
    maldi_view_test = myModel_mul.struct_data(maldi_view_test, method="reg", V=greg_maldi, kernel='linear')

    predictions_train = myModel_mul.predict([0], [1], maldi_view_train)
    predictions_test = myModel_mul.predict([0], [1], maldi_view_test)

    print("################ Results EXPERIMENT 1 ##################")
    print("auc IN SUBSET TRAIN")
    auc_carb = roc_auc_score(ramon_target[r_idx_train], predictions_train['output_view1']['mean_x'])
    print(auc_carb)
    auc_list_1.append(auc_carb)
    print("auc IN SUBSET TEST")
    auc_carb = roc_auc_score(ramon_target[r_idx_test], predictions_test['output_view1']['mean_x'])
    print(auc_carb)
    auc_list_2.append(auc_carb)

print(np.mean(auc_list_1))
print(np.std(auc_list_1))
print(np.mean(auc_list_2))
print(np.std(auc_list_2))

#### analizar resultados
#%%
from matplotlib import pyplot as plt
indicator_pos = 1
####################################### PLOT THE LATENT SPACE FEATURES BY VIEW ###################################3
# K dimension of the latent space
k_len = len(myModel_mul.q_dist.alpha[indicator_pos]['b'])
# Reorder the features w.r.t the more important ones for the RM view
ord_k = np.argsort(myModel_mul.q_dist.alpha[indicator_pos]['a']/myModel_mul.q_dist.alpha[indicator_pos]['b'])
# Create the matrix to plot afterwards
matrix_views = np.zeros((len(myModel_mul.q_dist.W), myModel_mul.q_dist.W[0]['mean'].shape[1]))
# There are 3 views
n_rowstoshow = 3
matrix_views = np.zeros((n_rowstoshow, myModel_mul.q_dist.W[0]['mean'].shape[1]))
for z in range(matrix_views.shape[0]):
    if z==0:
        X=myModel_mul.X[0]['X']
        # As the first view is kernel we have to go back to the primal space:
        W=X.T@myModel_mul.q_dist.W[0]['mean']
        matrix_views[z, :]=np.mean(np.abs(W), axis=0)[ord_k]
    else:
        matrix_views[z, :]=np.mean(np.abs(myModel_mul.q_dist.W[z]['mean']), axis=0)[ord_k]

titles_views = ['MALDI', 'HOSPITAL PROCEDENCE', 'OXA-48']

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
cmap = cm.get_cmap('Dark2', 9)
for v, view in enumerate(titles_views):
    if v==0: delay=9
    if v==1: delay=0
    if v==5: delay=1
    if v==7: delay=2
    if v==10: delay=3
    if v==11: delay=4
    if v==14: delay=5
    if v==15: delay=6
    if v==18: delay=7
    if v==19: delay=8
    plt.figure(figsize=[10,5])
    ax = plt.gca()
    ax.set_title(view+" view", color=cmap(delay))
    plt.yticks([], [])
    plt.xticks(range(0, len(ord_k)), ord_k.tolist())
    plt.xlabel("K features")
    im = ax.imshow(matrix_views[v,:][:, np.newaxis].T, cmap="binary")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # spath = "./Results/plots_reunion2406/"+view.replace(" ", "").replace("/","-")+"_19v_lfanalysis.png"
    # plt.savefig(spath)
    plt.show()

#%%
####################################### PLOT SOME FEATURES IN DETAIL ###################################3
X=myModel_mul.X[0]['X']
W_maldi=X.T@myModel_mul.q_dist.W[0]['mean']
# SELECT HERE WHICH FEATURES DO YOU WANT TO PLOT IN DETAIL!!!!!

# To see from which hospital they are select 1
# To see why they are oxas or not select 2
indicator_pos=1


lf = np.argsort(-np.abs(myModel_mul.q_dist.W[indicator_pos]['mean'])).squeeze()[:3]
# SOME STYLES
ls=['--', '-.', ':', (0, (1,1)), (0, (3,5,1,5))]

# PLOT ALL THE FEATURES AT THE SAME TIME
plt.figure(figsize=[15,10])
plt.title("Latent feature "+str(lf)+ "in MALDI weight matrix")
for k,l in enumerate(lf):
    alpha=0.5
    plt.plot(range(2000,12000), W_maldi[:, l]*myModel_mul.q_dist.W[1]['mean'][0,l], alpha=alpha, linestyle=ls[k], label="Latent feature "+str(l))
plt.legend()
plt.show()

#%% 
########################### CHECK ROIS
color=['tab:red', 'tab:green', 'tab:pink', 'tab:gray']

X=myModel_mul.X[0]['X']
W_maldi=X.T@myModel_mul.q_dist.W[0]['mean']
# TO CHECK HOSPITAL VIEW SELECT 1, TO CHECK OXA-48 SELECT 2
indicator_pos = 1

lf = np.argsort(-np.abs(myModel_mul.q_dist.W[indicator_pos]['mean'])).squeeze()[:3]

label1 = "Gregorio Marañón samples"
label2 = "Ramón y Cajal samples"

greg_oxas = np.mean(greg_maldi, axis=0)
ramon_oxas = np.mean(ramon_maldi, axis=0)

roi = [2450,2550]

plt.figure(figsize=[15,10])
plt.title("ZOOM IN: hospital view")
plt.plot(range(roi[0],roi[1]), greg_oxas[roi[0]-2000:roi[1]-2000], 'k--', label=label1)
plt.plot(range(roi[0],roi[1]), ramon_oxas[roi[0]-2000:roi[1]-2000], 'k', label=label2)
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*myModel_mul.q_dist.W[1]['mean'][0,l]/10000
    plt.plot(range(roi[0],roi[1]), W_proj[roi[0]-2000:roi[1]-2000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
# plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
plt.legend()
plt.grid()
plt.show()
#%%
################# Project to 3D
from sklearn.decomposition import PCA
indicator_pos = 1
pca = PCA(n_components=3)

### OPTION 1: USE THE 3 MOST RELEVANT FEATURES TO PROJECT
ord_k = np.argsort(-np.abs(myModel_mul.q_dist.W[indicator_pos]['mean'])).squeeze()
Z = myModel_mul.q_dist.Z['mean'][:, ord_k[:3].astype(int)]

### OPTION 2: PROJECT ALL THE FEATURES TO 3 FEATURES USING A PCA
# pca = PCA(n_components=3)
# relevant_views = np.mean(np.abs(myModel_mul.q_dist.W[indicator_pos]['mean']), axis=0)>1e-2
# Z = pca.fit_transform(myModel_mul.q_dist.Z['mean'][:, np.argwhere(relevant_views==1).squeeze()])

ramon_oxa_pos = np.argwhere((ramon_data['full']['CP gene']==1).values==1).squeeze()
ramon_indicator[ramon_oxa_pos, 0] = 2
indicator_values = np.vstack((greg_indicator, ramon_indicator))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
plt.title("Data projected over Z")
scatter1 = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=indicator_values)
plt.legend(handles=scatter1.legend_elements()[0], labels=["HGM OXA-48", "HRC OTHER", "HRC OXA-48"])
# legend1 = ax.legend(*scatter1.legend_elements(), loc='best', title="Classes")
# ax.add_artist(legend1)
ax.view_init(elev=10., azim=45)
# plt.legend()
ax.set_xlabel('Latent feature '+str(int(ord_k[0])))
ax.set_ylabel('Latent feature '+str(int(ord_k[1])))
ax.set_zlabel('Latent feature '+str(int(ord_k[2])))

plt.show()

#%%
# #%%
# #############################  EXPERIMENT 2

# ramon_index = ramon_data['full'][cols_ramon].index
# skf = StratifiedKFold(n_splits=5, shuffle=True)
# ramon_folds = {"train": [], "test": []}
# for train_index, test_index in skf.split(ramon_index, ramon_target):
#     ramon_folds["train"].append(ramon_index[train_index])
#     ramon_folds["test"].append(ramon_index[test_index])


# for f in range(5):
#     ramon_maldi_train = np.vstack(ramon_data['maldi'].loc[ramon_folds["train"][f]].values)
#     ramon_maldi_train /= np.mean(ramon_maldi_train)
#     ramon_maldi_test = np.vstack(ramon_data['maldi'].loc[ramon_folds["test"][f]].values)
#     ramon_maldi_test /= np.mean(ramon_maldi_train)

#     ramon_gen_train = ramon_data['full'][cols_ramon].loc[ramon_folds["train"][f]].values.astype(int)
#     ramon_gen_test = ramon_data['full'][cols_ramon].loc[ramon_folds["test"][f]].values.astype(int)

#     myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

#     MALDI_train = myModel_mul.struct_data(ramon_maldi_train, method="reg", V=ramon_maldi_train, kernel='linear')
#     GEN_train = myModel_mul.struct_data(ramon_gen_train, method="reg")

#     MALDI_test = myModel_mul.struct_data(ramon_maldi_test, method="reg", V=ramon_maldi_train, kernel='linear')
#     GEN_test = myModel_mul.struct_data(ramon_gen_test, method="reg")

#     ###### TRAIN MODEL
#     myModel_mul.fit(MALDI_train,
#                     GEN_train,
#                     max_iter=hyper_parameters['sshiba']['max_it'],
#                     pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
#                     verbose=1,
#                     feat_crit=1e-2)
    
#     predictions = myModel_mul.predict([0,1], [1], MALDI_test)
#     print("Media de las predicciones: "+str(np.mean(predictions["output_view1"]['mean_x'])))
#     print("STD de las predicciones: "+str(np.std(predictions["output_view1"]['mean_x'])))

#     auc_carb = roc_auc_score(ramon_gen_test, predictions["output_view1"]['mean_x'])
#     auc_exp1[0, f] = auc_carb
#     print(auc_carb)
# # %%

# %%
