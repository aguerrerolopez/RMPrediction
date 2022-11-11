
#%%
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
######################### PARAMETERS SELECTION ########################################
# Position where YOU introduced the Antibiotic Resistance view to the model during training
phen_pos = 2

# From which collection we want to see the results:
ryc = 1
hgm = 1

######################### LOAD DATA ########################################
if ryc:
    model_path = "./Results/normalizados/HRC_linear_normalizados_prun50.pkl"
    folds_path = "./data/RYC_5STRATIFIEDfolds_paper.pkl"
    data_path = "./data/ryc_data_paper.pkl"
if hgm: 
    model_path = "./Results/normalizados/HGM_linear_linpike_normalizados_prun50.pkl"
    folds_path = "./data/HGM_10STRATIFIEDfolds_paper.pkl"
    data_path = "./data/hgm_data_paper.pkl"

with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)
with open(model_path, 'rb') as pkl:
    results = pickle.load(pkl)

##### SELECT DATA
if hgm:
    old_fen = gm_data['fen']
    old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
    ph = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
    x = gm_data["maldi"].loc[ph.index]
if ryc:
    ph = gm_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
    x = gm_data['maldi'].loc[ph.index]

# CREATE RESULTS MATRIX
if hgm: n_fold = 10
if ryc: n_fold = 5
auc_by_phen = np.zeros((n_fold,3))


for f in range(len(folds["train"])):

    model_name = "model_fold" + str(f)

    # SELECT TRUE VALUES FOR VALIDATION DATA
    ph_val = ph.loc[folds["val"][f]].values
    ### PRED AR
    y_true = ph_val
    y_pred = results[model_name].t[phen_pos]['mean'][-ph_val.shape[0]:, :]
    print(np.std(y_pred))
    auc_by_phen[f,0] = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    auc_by_phen[f,1] = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    auc_by_phen[f,2] = roc_auc_score(y_true[:, 2], y_pred[:, 2])
    print(results[model_name].q_dist.W[0]['mean'].shape)
   
########## PRINT RESULTS IN MEAN AND STD W.R.T 10 FOLDS
print("RESULTS ANTIBIOTIC RESISTANCE")
print("MEAN")
print(np.mean(auc_by_phen, axis=0))
print("STD")
print(np.std(auc_by_phen, axis=0))


########### EXTRA RESULTS TAKE A LOOK IF YOU NEED THEM:
#%%
######################################### DETAIL RESULTS FOR THE FOLD WHOSE PERFORMANCE IS CLOSER TO THE MEAN #############################
representative_fold = np.argmin(np.abs(np.sum((auc_by_phen-np.mean(auc_by_phen, axis=0)), axis=1)))
f=representative_fold
model_name = "model_fold"+str(f)

####################################### PLOT THE LATENT SPACE FEATURES BY VIEW ###################################3
# K dimension of the latent space
k_len = len(results[model_name].q_dist.alpha[phen_pos]['b'])
# Reorder the features w.r.t the more important ones for the RM view
k_ords = np.zeros((10, k_len))
ord_k = np.argsort(results[model_name].q_dist.alpha[phen_pos]['a']/results[model_name].q_dist.alpha[phen_pos]['b'])
k_ords[f, :len(ord_k)] = ord_k
# Create the matrix to plot afterwards
matrix_views = np.zeros((len(results[model_name].q_dist.W),results[model_name].q_dist.W[0]['mean'].shape[1]))
# There are 13 views because the CARB+ESBL, ESBL and SUSCEPTIBLE are in the same view.
n_rowstoshow = 13
matrix_views = np.zeros((n_rowstoshow,results[model_name].q_dist.W[0]['mean'].shape[1]))
for z in range(matrix_views.shape[0]):
    if z==0:
        X=results[model_name].X[0]['X']
        # As the first view is kernel we have to go back to the primal space:
        W=X.T@results[model_name].q_dist.W[0]['mean']
        matrix_views[z, :]=np.mean(np.abs(W), axis=0)[ord_k]
    elif z>0 and z<4:
        matrix_views[z, :]=np.abs(results[model_name].q_dist.W[1]['mean'][z-1,:][ord_k])
    else:
        print(z-2)
        matrix_views[z, :]=np.mean(np.abs(results[model_name].q_dist.W[z-2]['mean']), axis=0)[ord_k]

titles_views = ['MALDI', 'CARB and ESBL', 'ESBL', 'Susceptible']

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
cmap = cm.get_cmap('Dark2', 4)
for v, view in enumerate(titles_views):
    plt.figure(figsize=[10,5])
    ax = plt.gca()
    ax.set_title(view+" view", color=cmap(v))
    plt.yticks([], [])
    plt.xticks(range(0, len(ord_k)), ord_k.tolist())
    plt.xlabel("K features")
    im = ax.imshow(matrix_views[v,:][:, np.newaxis].T, cmap="binary")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

#%%
representative_fold = np.argmin(np.abs(np.sum((auc_by_phen-np.mean(auc_by_phen, axis=0)), axis=1)))
f=representative_fold
model_name = "model_fold"+str(f)
####################################### PLOT SOME FEATURES IN DETAIL ###################################3
X=results[model_name].X[0]['X']
# Recover the primal space to check the important features.
W_maldi=X.T@results[model_name].q_dist.W[0]['mean']

# SELECT HERE WHICH VIEW DO YOU WANT TO USE TO ORDER THE RELEVANCE OF THE FEATURES
indicator_pos = 1

lf = np.argsort(-np.abs(results[model_name].q_dist.W[indicator_pos]['mean'])).squeeze()[:3]
# SOME STYLES
ls=['--', '-.', ':', (0, (1,1)), (0, (3,5,1,5))]

# PLOT ALL THE FEATURES AT THE SAME TIME
plt.figure(figsize=[15,10])
plt.title("Latent feature "+str(lf)+ "in MALDI weight matrix")
for k,l in enumerate(lf):
    alpha=0.5
    plt.plot(range(2000,12000), W_maldi[:, l], alpha=alpha, linestyle=ls[k], label="Latent feature "+str(l))
plt.legend()
plt.show()
