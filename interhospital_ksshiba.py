#%%
import pickle
import sys
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
import numpy as np
sys.path.insert(0, "./lib")
from lib import fast_fs_ksshiba_b_ord as ksshiba
import json
import telegram
from data import MaldiTofSpectrum
import topf
from sklearn.metrics import roc_auc_score

# Telegram bot
def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)


################## LOAD DATA ###################

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
################################# EXPERIMENT 2: PREDICT ANTIBIOTIC RESISTANCE in both hospitals with/without hospital indicator label########################################

###### PREPARE MODEL

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 50, "max_it": int(5000)}}
kernel = "linear"
kernel2 = "rbf"
weis_et_al=0
mkl = 1

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
    ## Indicator label
    x2_greg_train = np.zeros((x1_greg_train.shape[0], 1))
    x2_greg_test = np.zeros((x1_greg_test.shape[0], 1))

    # x2_greg_train = np.ones((x1_greg_train.shape[0], 5))
    # x2_greg_test = np.ones((x1_greg_test.shape[0], 5))
    # x2_greg_train[:, 2:4] = 0
    # x2_greg_test[:, 2:4] = 0

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
    ## Indicator label
    x2_ramon_train = np.ones((x1_ramon_train.shape[0], 1))
    x2_ramon_test = np.ones((x1_ramon_test.shape[0], 1))
    # x2_ramon_train = np.vstack(ramon_data['full']["Hospital"].loc[x1_tr.index])
    # x2_ramon_test = np.vstack(ramon_data['full']["Hospital"].loc[x1_val.index])

    ##### PREPARE VIEWS TRAIN/TEST
    maldi_view = np.vstack((x0_greg_train, x0_ramon_train, x0_greg_test, x0_ramon_test))
    indicator_view = np.vstack((x2_greg_train, x2_ramon_train, x2_greg_test, x2_ramon_test))
    rm_view = np.vstack((x1_greg_train, x1_ramon_train))


    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    if weis_et_al:
        x0_new = [[] for i in range(maldi_view.shape[0])]
        for asig in range(maldi_view.shape[0]):
            transformer= topf.PersistenceTransformer(n_peaks=400)
            print("Preproccessing TOPF "+str(asig)+"/"+str(maldi_view.shape[0]), end="\r")
            topf_signal = np.concatenate((np.arange(2000,12000).reshape(-1, 1), maldi_view[asig,:].reshape(-1,1)), axis=1)
            signal_transformed = transformer.fit_transform(topf_signal)
            x0_new[asig] = MaldiTofSpectrum(signal_transformed[signal_transformed[:,1]>0])
        maldi_view = [MaldiTofSpectrum(x0_new[i]) for i in range(len(x0_new))]
        maldi_view = myModel_mul.struct_data(maldi_view, method="reg", V=maldi_view, kernel="pike")
    else:
        if mkl:
            maldi_view1 = myModel_mul.struct_data(maldi_view, method="reg", V=np.vstack((x0_greg_train, x0_ramon_train)), kernel=kernel)
            maldi_view2 = myModel_mul.struct_data(maldi_view, method="reg", V=np.vstack((x0_greg_train, x0_ramon_train)), kernel=kernel2)
        else:
            maldi_view = myModel_mul.struct_data(maldi_view, method="reg", V=np.vstack((x0_greg_train, x0_ramon_train)), kernel=kernel)
    
    rm_view = myModel_mul.struct_data(rm_view, method="mult")
    indicator_view = myModel_mul.struct_data(indicator_view, method="mult")

    myModel_mul.fit(maldi_view1, maldi_view2, 
                    #indicator_view, ############# UNCOMMENT THIS LINE IF YOU WANT TO ADD THE EXTRA HOSPITAL OF ORIGIN VIEW
                    rm_view,
                    max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1,
                    feat_crit=1e-2)

    print("################ Results EXPERIMENT Resistant Mechanisms in both hospitals##################")

    ####### DO THE PREDICTION
    n_ramon = x1_ramon_test.shape[0]
    n_greg = x1_greg_test.shape[0]+n_ramon

    greg_prediction = myModel_mul.t[2]['mean'][-n_greg:-n_ramon,:]
    ramon_prediction = myModel_mul.t[2]['mean'][-n_ramon:,:]

    
    for res in range(3):
        # PREDICT ONLY IN GREG
        auc_carb = roc_auc_score(x1_greg_test[:, res], greg_prediction[:, res])
        greg[res].append(auc_carb)
        print(x1_greg_test[:, res])
        print(greg_prediction[:, res])
        print(auc_carb)
        # PREDICT ONLY IN RAMON
        auc_carb = roc_auc_score(x1_ramon_test[:, res], ramon_prediction[:, res])
        ramon[res].append(auc_carb)

print("######### RESULTS")

print("AUC in Greg hospital")
print("CP+ESBL: "+str(np.mean(greg[0]))+"+-"+str(np.std(greg[0])))
print("ESBL: "+str(np.mean(greg[1]))+"+-"+str(np.std(greg[1])))
print("S: "+str(np.mean(greg[2]))+"+-"+str(np.std(greg[2])))

print("AUC in HRC hospital")
print("CP+ESBL: "+str(np.mean(ramon[0]))+"+-"+str(np.std(ramon[0])))
print("ESBL: "+str(np.mean(ramon[1]))+"+-"+str(np.std(ramon[1])))
print("S: "+str(np.mean(ramon[2]))+"+-"+str(np.std(ramon[2])))

#### ANALYZE RESULTS
#%%
import pickle
with open('both_hospital_latentfeatureanalysis.pkl', 'rb') as handle:
    b = pickle.load(handle)
X = b['X']
myModel_mul = b["model"]

from matplotlib import pyplot as plt
indicator_pos = 2
####################################### PLOT THE LATENT SPACE FEATURES BY VIEW ###################################3
# K dimension of the latent space
k_len = len(myModel_mul.q_dist.alpha[indicator_pos]['b'])
# Reorder the features w.r.t the more important ones for the RM view
ord_k = np.argsort(myModel_mul.q_dist.alpha[indicator_pos]['a']/myModel_mul.q_dist.alpha[indicator_pos]['b'])
# Create the matrix to plot afterwards
matrix_views = np.zeros((len(myModel_mul.q_dist.W), myModel_mul.q_dist.W[0]['mean'].shape[1]))
# There are 3 views
n_rowstoshow = 3
matrix_views = np.zeros((n_rowstoshow+2, myModel_mul.q_dist.W[0]['mean'].shape[1]))
for z in range(n_rowstoshow):
    if z==0:
        # As the first view is kernel we have to go back to the primal space:
        W=X.T@myModel_mul.q_dist.W[0]['mean']
        matrix_views[z, :]=np.mean(np.abs(W), axis=0)[ord_k]
    if z>1:
        matrix_views[z, :]=np.abs(myModel_mul.q_dist.W[z]['mean'][0,:])[ord_k]
        matrix_views[z+1, :]=np.abs(myModel_mul.q_dist.W[z]['mean'][1,:])[ord_k]
        matrix_views[z+2, :]=np.abs(myModel_mul.q_dist.W[z]['mean'][2,:])[ord_k]
    else:
        matrix_views[z, :]=np.mean(np.abs(myModel_mul.q_dist.W[z]['mean']), axis=0)[ord_k]

titles_views = ['MALDI-TOF MS', 'HOSPITAL OF ORIGIN', 'ESBL+CP', 'ESBL', 'S']

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
cmap = cm.get_cmap('Dark2', 9)
fig, ax = plt.subplots(3,1, figsize=(30,10), gridspec_kw={'height_ratios': [5, 5, 5]})
plt.setp(ax, xticks=range(0,len(ord_k)), xticklabels=[], yticks=[])
order=[2,0,1]
for v, o in enumerate(order):
    if o==2:
        ax[v].set_title("Resistance Mechanism view", color=cmap(v), fontsize=30)
        ax[v].text(x=20, y=0, s="CARB+ESBL", fontsize=16,horizontalalignment='center', verticalalignment='center', )
        ax[v].text(x=20, y=1, s="ESBL", fontsize=16,horizontalalignment='center', verticalalignment='center')
        ax[v].text(x=20, y=2, s="Susceptible",fontsize=16,horizontalalignment='center', verticalalignment='center')
    else: ax[v].set_title(titles_views[o]+" view", color=cmap(v), fontsize=20)
    ax[v].set_xlabel("K features", fontsize=24)
    # ax[v].tick_params(axis="x", labelsize=16) 
    if o==2:
        im = ax[v].imshow(matrix_views[o:,:], cmap="binary")
    else:
        im = ax[v].imshow(matrix_views[o,:][:, np.newaxis].T, cmap="binary")
    divider = make_axes_locatable(ax[v])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    # ax[v].colorbar(im)
    # spath = "./Results/plots_reunion2406/"+view.replace(" ", "").replace("/","-")+"_19v_lfanalysis.png"
    # plt.savefig(spath)
plt.show()

lf = [59, 46, 38]
plt.figure()
for l in lf:
    W_toplot = W[:, l]*myModel_mul.q_dist.W[1]['mean'][0,l]/1e4
    plt.plot(range(2000,12000), W_toplot, label="Latent feature "+str(l))
plt.legend()
plt.show()

# #%%
# ####################################### PLOT SOME FEATURES IN DETAIL ###################################3
# X=myModel_mul.X[0]['X']
# W_maldi=X.T@myModel_mul.q_dist.W[0]['mean']
# # SELECT HERE WHICH FEATURES DO YOU WANT TO PLOT IN DETAIL!!!!!

# # To see from which hospital they are select 1
# # To see why they are oxas or not select 2
# indicator_pos=1

# greg_oxa = np.vstack((greg_target[g_idx_train], greg_target[g_idx_test]))
# greg_samples_oxas = np.mean(greg_maldi[np.where(greg_oxa==1)[0],:], axis=0)
# greg_samples_nooxas = np.mean(greg_maldi[np.where(greg_oxa==0)[0],:], axis=0)


# ramon_oxa = np.vstack((ramon_target[r_idx_train], ramon_target[r_idx_test]))
# ramon_samples_oxas = np.mean(ramon_maldi[np.where(ramon_oxa==1)[0],:], axis=0)
# ramon_samples_nooxas = np.mean(ramon_maldi[np.where(ramon_oxa==0)[0],:], axis=0)


# lf = np.argsort(-np.abs(myModel_mul.q_dist.W[indicator_pos]['mean'])).squeeze()[:3]
# # SOME STYLES
# ls=['--', '-.', ':', (0, (1,1)), (0, (3,5,1,5))]

# # PLOT ALL THE FEATURES AT THE SAME TIME
# plt.figure(figsize=[15,10])
# plt.title("Latent feature "+str(lf)+ "in MALDI weight matrix")
# for k,l in enumerate(lf):
#     alpha=0.5
#     plt.plot(range(2000,12000), W_maldi[:, l]*myModel_mul.q_dist.W[1]['mean'][0,l]/10000, alpha=alpha, linestyle=ls[k], label="Latent feature "+str(l))
# # plt.plot(range(2000,12000), greg_oxas, label="Gregorio samples")
# plt.plot(range(2000,12000), ramon_samples_oxas, label="Ramon OXA samples")
# plt.plot(range(2000,12000), ramon_samples_nooxas, label="Ramon other samples")
# plt.legend()
# plt.show()

# #%% 
# from matplotlib import pyplot as plt
# ########################### CHECK ROIS
# color=['tab:red', 'tab:green', 'tab:pink', 'tab:gray']

# X=myModel_mul.X[0]['X']
# W_maldi=X.T@myModel_mul.q_dist.W[0]['mean']
# # TO CHECK HOSPITAL VIEW SELECT 1, TO CHECK OXA-48 SELECT 2
# indicator_pos = 2

# lf = np.argsort(-np.abs(myModel_mul.q_dist.W[indicator_pos]['mean'])).squeeze()[:3]

# label1 = "Mean Gregorio Marañón OXA samples"
# label2 = "Mean Gregorio other samples"
# label3 = "Mean Ramón y Cajal OXA samples"
# label4 = "Mean Ramón y Cajal other samples"

# greg_oxa = np.vstack((greg_target[g_idx_train], greg_target[g_idx_test]))
# greg_samples_oxas = np.mean(greg_maldi[np.where(greg_oxa==1)[0],:], axis=0)
# greg_samples_nooxas = np.mean(greg_maldi[np.where(greg_oxa==0)[0],:], axis=0)


# ramon_oxa = np.vstack((ramon_target[r_idx_train], ramon_target[r_idx_test]))
# ramon_samples_oxas = np.mean(ramon_maldi[np.where(ramon_oxa==1)[0],:], axis=0)
# ramon_samples_nooxas = np.mean(ramon_maldi[np.where(ramon_oxa==0)[0],:], axis=0)

# roi = [9100,9300]

# plt.figure(figsize=[15,10])
# plt.title("ZOOM IN: oxa view")
# plt.plot(range(roi[0],roi[1]), greg_samples_oxas[roi[0]-2000:roi[1]-2000], 'k--', label=label1)
# plt.plot(range(roi[0],roi[1]), greg_samples_nooxas[roi[0]-2000:roi[1]-2000], 'r--', label=label2)

# plt.plot(range(roi[0],roi[1]), ramon_samples_oxas[roi[0]-2000:roi[1]-2000], 'k', label=label3)
# plt.plot(range(roi[0],roi[1]), ramon_samples_nooxas[roi[0]-2000:roi[1]-2000], 'r', label=label4)
# for j,l in enumerate(lf):
#     W_proj = W_maldi[:, l]*myModel_mul.q_dist.W[1]['mean'][0,l]/10000
#     plt.plot(range(roi[0],roi[1]), W_proj[roi[0]-2000:roi[1]-2000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
# # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
# plt.legend()
# plt.grid()
# plt.show()
# #%%
# ################# Project to 3D
# from sklearn.decomposition import PCA
# indicator_pos = 1
# pca = PCA(n_components=3)

# ### OPTION 1: USE THE 3 MOST RELEVANT FEATURES TO PROJECT
# ord_k = np.argsort(-np.abs(myModel_mul.q_dist.W[indicator_pos]['mean'])).squeeze()
# Z = myModel_mul.q_dist.Z['mean'][:, ord_k[:3].astype(int)]

# ### OPTION 2: PROJECT ALL THE FEATURES TO 3 FEATURES USING A PCA
# # pca = PCA(n_components=3)
# # relevant_views = np.mean(np.abs(myModel_mul.q_dist.W[indicator_pos]['mean']), axis=0)>1e-2
# # Z = pca.fit_transform(myModel_mul.q_dist.Z['mean'][:, np.argwhere(relevant_views==1).squeeze()])

# ramon_oxa_pos = np.argwhere((ramon_data['full']['CP gene']==1).values==1).squeeze()
# ramon_indicator[ramon_oxa_pos, 0] = 2
# indicator_values = np.vstack((greg_indicator, ramon_indicator))

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111, projection='3d')
# plt.title("Data projected over Z")
# scatter1 = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=indicator_values)
# plt.legend(handles=scatter1.legend_elements()[0], labels=["HGM OXA-48", "HRC OTHER", "HRC OXA-48"])
# # legend1 = ax.legend(*scatter1.legend_elements(), loc='best', title="Classes")
# # ax.add_artist(legend1)
# ax.view_init(elev=10., azim=45)
# # plt.legend()
# ax.set_xlabel('Latent feature '+str(int(ord_k[0])))
# ax.set_ylabel('Latent feature '+str(int(ord_k[1])))
# ax.set_zlabel('Latent feature '+str(int(ord_k[2])))

# plt.show()

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
