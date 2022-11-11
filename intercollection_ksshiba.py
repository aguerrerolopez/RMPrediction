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

# Telegram bot: I have a telegram bot that sends me a message whenever the training has finished. Just dont pay attention to this if you dont want to use it.
def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)


################## LOAD DATA ###################
data_path = "./data/ryc_data_paper.pkl"
with open(data_path, 'rb') as pkl:
    ramon_data = pickle.load(pkl)
data_path = "./data/gm_data_paper.pkl"
with open(data_path, 'rb') as pkl:
    greg_data = pickle.load(pkl)

fen_greg = greg_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
# fen_greg = greg_data['full'][['CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
maldi_greg = greg_data['maldi'].loc[fen_greg.index]
fen_ramon = ramon_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
maldi_ramon = ramon_data['maldi'].loc[fen_ramon.index]

#%%

##################### PARAMETERS SELECTION ####################
# Prune = to prune the latent space
# myKc = initial K dimension of latent space
# pruning crit to prune K dimensions
# max_it = maximum iterations to wait until convergence
hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-2, "max_it": int(5000)}}
# KERNEL SELECTION: pike, linear or rbf
kernel = "linear" 


###### LOAD FOLDS
folds_path = "./data/GM_5STRATIFIEDfolds_paper.pkl"
with open(folds_path, 'rb') as pkl:
    folds_greg = pickle.load(pkl)

folds_path = "./data/RYC_5STRATIFIEDfolds_paper.pkl"
with open(folds_path, 'rb') as pkl:
    folds_ramon = pickle.load(pkl) 

greg = [[], [], []]
ramon = [[], [], []]
sig = []
##################### TRAIN MODEL AND PREDICT ######################
for r in range(5):
    print("Training model "+str(r))
    # Select train and test data

    ##### GM data
    ## MALDI data.
    x0_tr, x0_val = maldi_greg.loc[folds_greg["train"][r]], maldi_greg.loc[folds_greg["val"][r]]
    x0_greg_train= np.vstack(x0_tr.values).astype(float)
    x0_greg_test = np.vstack(x0_val.values).astype(float)
    x0_greg_test /= np.mean(x0_greg_train)
    x0_greg_train /= np.mean(x0_greg_train)

    # AST
    x1_tr, x1_val = fen_greg.loc[folds_greg["train"][r]], fen_greg.loc[folds_greg["val"][r]]
    x1_greg_train = np.vstack(x1_tr.values)
    x1_greg_test = np.vstack(x1_val.values)

    # HOSPITAL COLLECTION OF ORIGIN data
    x2_greg_train = np.zeros((x1_greg_train.shape[0], 1))
    x2_greg_test = np.zeros((x1_greg_test.shape[0], 1))

    ##### RyC data
    # AST
    x1_tr, x1_val = fen_ramon.loc[folds_ramon["train"][r]].dropna(), fen_ramon.loc[folds_ramon["val"][r]].dropna()
    x1_ramon_train = np.vstack(x1_tr.values)
    x1_ramon_test = np.vstack(x1_val.values)

    # MALDI data
    x0_tr, x0_val = maldi_ramon.loc[x1_tr.index], maldi_ramon.loc[x1_val.index]
    x0_ramon_train= np.vstack(x0_tr.values).astype(float)
    x0_ramon_test = np.vstack(x0_val.values).astype(float)
    x0_ramon_test /= np.mean(x0_ramon_train)
    x0_ramon_train /= np.mean(x0_ramon_train)
    
    # HOSPITAL COLLECTION OF ORIGIN data
    x2_ramon_train = np.ones((x1_ramon_train.shape[0], 1))
    x2_ramon_test = np.ones((x1_ramon_test.shape[0], 1))
    # In case you want to differ between HOSPITALS from where the sample are instead of CENTRE which studied the AST do this:
    # x2_ramon_train = np.vstack(ramon_data['full']["Hospital"].loc[x1_tr.index])
    # x2_ramon_test = np.vstack(ramon_data['full']["Hospital"].loc[x1_val.index])

    ##### PREPARE VIEWS TRAIN/TEST
    maldi_view = np.vstack((x0_greg_train, x0_ramon_train, x0_greg_test, x0_ramon_test))
    indicator_view = np.vstack((x2_greg_train, x2_ramon_train, x2_greg_test, x2_ramon_test))
    # The ones that you want to predict are the ones that you DO NOT INCLUDE. The model will take them as "missing" and will predict them.
    ast_view = np.vstack((x1_greg_train, x1_ramon_train))

    ##### Declare the data as the model needs it:
    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    # If you want to use the PIKE kernel, the data has to be transformed into their required shape:
    if kernel=="pike":
        x0_new = [[] for i in range(maldi_view.shape[0])]
        for asig in range(maldi_view.shape[0]):
            transformer= topf.PersistenceTransformer(n_peaks=400)
            print("Preproccessing TOPF "+str(asig)+"/"+str(maldi_view.shape[0]), end="\r")
            topf_signal = np.concatenate((np.arange(2000,12000).reshape(-1, 1), maldi_view[asig,:].reshape(-1,1)), axis=1)
            signal_transformed = transformer.fit_transform(topf_signal)
            x0_new[asig] = MaldiTofSpectrum(signal_transformed[signal_transformed[:,1]>0])
        maldi_view = [MaldiTofSpectrum(x0_new[i]) for i in range(len(x0_new))]
        maldi_view = myModel_mul.struct_data(maldi_view, method="reg", V=maldi_view, kernel="pike")
    elif kernel=="rbf":
        maldi_view = myModel_mul.struct_data(maldi_view, method="reg", V=maldi_view, kernel="rbf")
    elif kernel=="linear":
        maldi_view = myModel_mul.struct_data(maldi_view, method="reg", V=maldi_view, kernel="linear")
    else: 
        print("You didn't choose a kernel, so we are going to use the MALDI data as raw:")
        maldi_view = myModel_mul.struct_data(maldi_view, method="reg")
    
    # Antibiotic Resistance mechanisms view: a multilabel view
    print(ast_view.shape)
    ast_view = myModel_mul.struct_data(ast_view, method="mult")
    # HOSPITAL COLLECTION OF ORIGIN view: a multilabel view (binary)
    indicator_view = myModel_mul.struct_data(indicator_view, method="mult")

    ##################### TRAIN THE MODEL ###################### 
    myModel_mul.fit(maldi_view,
                    indicator_view, ############# UNCOMMENT THIS LINE IF YOU WANT TO ADD THE EXTRA HOSPITAL COLLECTION OF ORIGIN VIEW
                    ast_view,
                    max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1,
                    feat_crit=1e-2)
    sig.append(myModel_mul.sig[0])
    print("################ Results EXPERIMENT Resistant Mechanisms in both hospitals##################")

    ####### DO THE PREDICTION: the model has predicted it automatically because it found it missing. Therefore, the prediction is located
    # where the missing points where located.
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
print("AUC in GM collection")
print("CP+ESBL: "+str(np.mean(greg[0]))+"+-"+str(np.std(greg[0])))
print("ESBL: "+str(np.mean(greg[1]))+"+-"+str(np.std(greg[1])))
print("WT: "+str(np.mean(greg[2]))+"+-"+str(np.std(greg[2])))

print("AUC in RyC collection")
print("CP+ESBL: "+str(np.mean(ramon[0]))+"+-"+str(np.std(ramon[0])))
print("ESBL: "+str(np.mean(ramon[1]))+"+-"+str(np.std(ramon[1])))
print("WT: "+str(np.mean(ramon[2]))+"+-"+str(np.std(ramon[2])))

print("SIGMA value")
print(np.mean(sig))

### ANALYZE RESULTS: here we plotted the latent space, check it if you want to plot similar results.
#%%
#import pickle
#with open('./Results/both_hospital_latentfeatureanalysis.pkl', 'rb') as handle:
#    b = pickle.load(handle)
#X = b['X']
#myModel_mul = b["model"]

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
matrix_views = np.zeros((n_rowstoshow, myModel_mul.q_dist.W[0]['mean'].shape[1]))
for z in range(n_rowstoshow):
    if z==0:
        # As the first view is kernel we have to go back to the primal space:
        W=X.T@myModel_mul.q_dist.W[0]['mean']
        matrix_views[z, :]=np.mean(np.abs(W), axis=0)[ord_k]
    if z>1:
        matrix_views[z, :]=np.abs(myModel_mul.q_dist.W[z]['mean'][0,:])[ord_k]
    else:
        matrix_views[z, :]=np.mean(np.abs(myModel_mul.q_dist.W[z]['mean']), axis=0)[ord_k]

titles_views = ['MALDI-TOF MS', 'DOMAIN', 'WT', 'ESBL', 'ESBL+CP']

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
cmap = cm.get_cmap('Dark2', 9)
fig, ax = plt.subplots(3,1, figsize=(30,10), gridspec_kw={'height_ratios': [5, 5, 5]})
plt.setp(ax, xticks=range(0,len(ord_k)), xticklabels=[], yticks=[])
order=[2,0,1]
for v, o in enumerate(order):
    if o==2:
        ax[v].set_title("ANTIBIOTIC RESISTANCE view", color=cmap(v), fontsize=30)
        ax[v].text(x=20, y=2, s="Wild Type",fontsize=16,horizontalalignment='center', verticalalignment='center')
        ax[v].text(x=20, y=1, s="ESBL", fontsize=16,horizontalalignment='center', verticalalignment='center')
        ax[v].text(x=20, y=0, s="ESBL+CP", fontsize=16,horizontalalignment='center', verticalalignment='center')
    else: ax[v].set_title(titles_views[o]+" view", color=cmap(v), fontsize=30)
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

# %%
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
 
tsne = TSNE(n_components=2, random_state=10)
pca = PCA(n_components=2, random_state=42)

z_tsne_hospital = tsne.fit_transform(myModel_mul.q_dist.Z['mean'].squeeze())
z_pca_hospital = pca.fit_transform(myModel_mul.q_dist.Z['mean'].squeeze())
#z_tsne_hospital = tsne.fit_transform(myModel_mul.q_dist.Z['mean'][:,np.where(np.abs(myModel_mul.q_dist.W[1]['mean'][0,:])>0.01)].squeeze())
#z_pca_hospital = pca.fit_transform(myModel_mul.q_dist.Z['mean'][:,np.where(np.abs(myModel_mul.q_dist.W[1]['mean'][0,:])>0.01)].squeeze())

colors = ['tab:red','tab:green']
hospital_label = indicator_view['data'].squeeze().tolist()

first_ryc = hospital_label.index(1)
hospitals = np.vstack(ramon_data['full'].loc[folds_ramon["train"][r]]['Hospital'])
samples_start = x0_greg_train.shape[0]
samples_end= x0_greg_train.shape[0]+np.vstack(ramon_data['full'].loc[folds_ramon["train"][r]]['Hospital']).shape[0]
hospitals = ['GM', 'RyC']
ast =  ['ESBL+CP', 'ESBL', 'WT']

plt.figure(figsize=[15,10])
plt.title("TSNE projection of Z by domain view", fontsize=20)
scatter1 = plt.scatter(z_tsne_hospital[:, 0][:first_ryc], z_tsne_hospital[:, 1][:first_ryc], c='tab:red', marker='x', label="GM domain")
scatter2 = plt.scatter(z_tsne_hospital[:, 0][first_ryc:], z_tsne_hospital[:, 1][first_ryc:], c='tab:blue', marker='o', label="RyC domain")
plt.legend(fontsize=20)
plt.savefig('tsne_z_byhospital_withoutlabel.png')
plt.show()

plt.title("PCA projection of Z by domain")
scatter1 = plt.scatter(z_pca_hospital[:, 0][:first_ryc], z_pca_hospital[:, 1][:first_ryc], c='tab:red', marker='x', label="GM domain")
scatter2 = plt.scatter(z_pca_hospital[:, 0][first_ryc:], z_pca_hospital[:, 1][first_ryc:], c='tab:blue', marker='o', label="RyC domain")
plt.legend()
plt.show()


# %%
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

tsne = TSNE(n_components=3, random_state=10)
pca = PCA(n_components=3, random_state=42)

z_tsne_ab = tsne.fit_transform(myModel_mul.q_dist.Z['mean'][:,np.where(np.abs(myModel_mul.q_dist.W[2]['mean'][0,:])>0.01)].squeeze())
z_pca_ab = pca.fit_transform(myModel_mul.q_dist.Z['mean'][:,np.where(np.abs(myModel_mul.q_dist.W[2]['mean'][0,:])>0.01)].squeeze())

colors = ['tab:red','tab:green']
hospital_label = myModel_mul.t[1]['data'].squeeze().tolist()
ast_label = np.argmax(np.vstack((x1_greg_train, x1_ramon_train, x1_greg_test, x1_ramon_test)), axis=1)

first_ryc = hospital_label.index(1)
hospitals = ['GM', 'RyC']
ast =  ['ESBL+CP', 'ESBL', 'WT']

fig = plt.figure(figsize=[15,10])
plt.title("TSNE projection of Z by AR")
ax = fig.add_subplot(111, projection='3d')
scatter1 = ax.scatter(z_tsne_ab[:, 0][np.where(ast_label==0)], z_tsne_ab[:, 1][np.where(ast_label==0)], z_tsne_ab[:, 2][np.where(ast_label==0)], c='tab:red', marker='x', label="ESBL+CP")
scatter2 = ax.scatter(z_tsne_ab[:, 0][np.where(ast_label==1)], z_tsne_ab[:, 1][np.where(ast_label==1)], z_tsne_ab[:, 2][np.where(ast_label==1)], c='tab:blue', marker='o', label="ESBL")
scatter3 = ax.scatter(z_tsne_ab[:, 0][np.where(ast_label==2)], z_tsne_ab[:, 1][np.where(ast_label==2)], z_tsne_ab[:, 2][np.where(ast_label==2)], c='tab:green', marker='s', label="WT")
plt.legend()
plt.show()

plt.title("PCA projection of Z by AR")
scatter1 = plt.scatter(z_pca_ab[:, 0][np.where(ast_label==0)], z_pca_ab[:, 1][np.where(ast_label==0)], z_pca_ab[:, 2][np.where(ast_label==0)], c='tab:red', marker='x', label="ESBL+CP")
scatter1 = plt.scatter(z_pca_ab[:, 0][np.where(ast_label==1)], z_pca_ab[:, 1][np.where(ast_label==1)], z_pca_ab[:, 2][np.where(ast_label==1)], c='tab:blue', marker='o', label="ESBL")
scatter1 = plt.scatter(z_pca_ab[:, 0][np.where(ast_label==2)], z_pca_ab[:, 1][np.where(ast_label==2)], z_pca_ab[:, 2][np.where(ast_label==2)], c='tab:green', marker='s', label="WT")
plt.legend()
plt.show()
# %%
