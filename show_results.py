
#%%
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
######################### PARAMETERS SELECTION ########################################
# Position where YOU introduced the RM view to the model during training
phen_pos = 2
ryc = 0
hgm = 1

######################### LOAD DATA ########################################
if ryc:
    model_path = "./Results/normalizados/HRC_mkl2kernels_linpike_normalizados_prun50.pkl"
    folds_path = "./data/RYC_5STRATIFIEDfolds_muestrascompensada_experimentAug.pkl"
    data_path = "./data/ryc_data_expAug.pkl"
if hgm: 
    model_path = "./Results/normalizados/HGM_mkl2kernels_linpike_normalizados_prun50.pkl"
    folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_pruebaloca.pkl"
    data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"

with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)
with open(model_path, 'rb') as pkl:
    results = pickle.load(pkl)

##### PREDICT CARB AND ESBL
if hgm:
    old_fen = gm_data['fen']
    old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
    ph = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
    x = gm_data["maldi"].loc[ph.index]
    y = gm_data['binary_ab'].loc[ph.index]
if ryc:
    ######### PREDICT CARB AND ESBL
    ph = gm_data['full'][['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
    x = gm_data['maldi'].loc[ph.index]
    y = gm_data['full'][['AMOXI/CLAV .1', 'PIP/TAZO.1', 'CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1', 'AZTREONAM.1', 'IMIPENEM.1']].loc[ph.index]



###### PREDICT OTHER AB
# x = gm_data["maldi"]
# y = gm_data['binary_ab']

if hgm: n_fold = 10; n_ab = 9
if ryc: n_fold = 5; n_ab = 7
auc_by_ab = np.zeros((n_fold,n_ab))
auc_by_phen = np.zeros((n_fold,3))


familias = { "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
                    "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
                    "monobactams": ['AZTREONAM'],
                    "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
                    "fluoro":['CIPROFLOXACINO'],
                    "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA'],
                    "fosfo": ['FOSFOMICINA'],
                    "otros":['COLISTINA']
                    }

for f in range(len(folds["train"])):
    y_val = y.loc[folds["val"][f]]
    ##### PREDICT CARB AND ESBL
    ph_val = ph.loc[folds["val"][f]].values
    y_val = y_val.values
    ###### PREDICT OTHER AB
    # cols = ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA', 'CIPROFLOXACINO', 'COLISTINA']
    # y_val = y_val[cols].values

    model_name = "model_fold" + str(f)

    # for h in range(7):
    #     ##### PREDICT CARB AND ESBL
    #     y_true = y_val[:, h]
    #     y_pred = results[model_name].t[2+h]['mean'][-y_val.shape[0]:, 0]
    # #     ###### PREDICT OTHER AB
    # #     # y_pred = results[model_name].t[1+h]['mean'][-y_val.shape[0]:, 0]
    #     auc_by_ab[f, h] = roc_auc_score(y_true, y_pred)
    #     # if h>5:
    #     #     break

    ### PRED RM
    y_true = ph_val
    y_pred = results[model_name].t[phen_pos]['mean'][-y_val.shape[0]:, :]
    print(np.std(y_pred))
    auc_by_phen[f,0] = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    auc_by_phen[f,1] = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    auc_by_phen[f,2] = roc_auc_score(y_true[:, 2], y_pred[:, 2])
    print(results[model_name].q_dist.W[0]['mean'].shape)
   
########## PRINT RESULTS IN MEAN AND STD W.R.T 10 FOLDS
print("RESULTS PHEN")
print("MEAN")
print(np.mean(auc_by_phen, axis=0))
print("STD")
print(np.std(auc_by_phen, axis=0))

print("RESULTS ANTIBIOTIC")
print("MEAN")
print(np.mean(auc_by_ab, axis=0))
print(np.mean(auc_by_ab))
print("STD")
print(np.std(auc_by_ab, axis=0))

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

titles_views = ['MALDI', 'CARB and ESBL', 'ESBL', 'Susceptible','AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM']

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
representative_fold = np.argmin(np.abs(np.sum((auc_by_phen-np.mean(auc_by_phen, axis=0)), axis=1)))
f=representative_fold
model_name = "model_fold"+str(f)
####################################### PLOT SOME FEATURES IN DETAIL ###################################3
X=results[model_name].X[0]['X']
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

#%%
# PLOT MEAN OF SENSIBLE AND RESISTANT SAMPLES TO THE 3 RM TASKS
titles=['to CARB', 'to ESBL']
for k, col in enumerate(ph.columns):
    plt.figure(figsize=[15,10])
    plt.title(col+" view")
    if k==0:
        label1 ="Resistant "+titles[0]+" or "+titles[1]
        label2 ="Sensible "+titles[0]+" and "+titles[1]
    if k==1:
        label1 ="Resistant "+titles[1]
        label2 ="Sensible "+titles[1]
    if k==2:
        label2 ="Sensible "+titles[0]+" and "+titles[1]
        label1 ="Resistant "+titles[0]+" or "+titles[1]
    r_samples = np.vstack(x.values[(ph[col]==1).values])
    s_samples = np.vstack(x.values[(ph[col]==0).values])
    plt.plot(range(2000,12000), np.mean(r_samples, axis=0), color='tab:blue', label=label1)
    plt.fill_between(range(2000,12000), np.mean(r_samples, axis=0)-np.std(r_samples, axis=0), np.mean(r_samples, axis=0)+np.std(r_samples, axis=0), color='tab:blue', alpha=0.2)
    plt.plot(range(2000,12000), np.mean(s_samples, axis=0), color='tab:red', label=label2)
    plt.fill_between(range(2000,12000), np.mean(s_samples, axis=0)-np.std(s_samples, axis=0), np.mean(s_samples, axis=0)+np.std(s_samples, axis=0), color='tab:red', alpha=0.2)
    plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend() 
    plt.show()


#%%
# PLOT THE W MATRIX ASSOCIATED TO EACH ONE OF THE 3 RM VIEWS
plt.rcParams.update({'font.size': 22})
X=results[model_name].X[0]['X']
W_maldi=X.T@results[model_name].q_dist.W[0]['mean']

lf = [np.argsort(-np.abs(results[model_name].q_dist.W[1]['mean'])).squeeze()[0, :3],
np.argsort(-np.abs(results[model_name].q_dist.W[1]['mean'])).squeeze()[1, :3],
np.argsort(-np.abs(results[model_name].q_dist.W[1]['mean'])).squeeze()[2, :3]]

color=['tab:red', 'tab:green', 'tab:pink', 'tab:gray']

res_samples = [np.vstack(x[ph[col]==1].sample(1, random_state=0).values)[0] for col in ph.columns]
sen_samples = [np.vstack(x[ph[col]==0].sample(1, random_state=0).values)[0] for col in ph.columns]

titulo = ['CARB and ESBL', 'ESBL', 'Susceptible']
for k, col in enumerate(ph.columns):
    plt.figure(figsize=[15,10])
    plt.title(titulo[k]+" view")
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.plot(range(2000,12000), res_samples[k], '--', alpha=0.2, label=label1)
    plt.plot(range(2000,12000), sen_samples[k], '--', alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(2000,12000), W_proj, color=color[j], alpha=0.8, label="Latent feature "+str(l))
    plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    plt.grid()
    path = "Results/plots_reunion2406/privadas_"+col+".png"
    # plt.savefig(path)
    plt.show()

#%%
# ###ZOOM IN ROI 1### PLOT THE W MATRIX ASSOCIATED TO EACH ONE OF THE 3 RM VIEWS 

zone1 = [2000,3000]
zone2 = [7000,8000]
zone3 = [8000,9000]
zone4 = [9000,10000]

for k, col in enumerate(ph.columns):
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.plot(range(zone1[0],zone1[1]), res_samples[k][zone1[0]-2000:zone1[1]-2000]*1e4, 'k--',  label=label1)
    plt.plot(range(zone1[0],zone1[1]), sen_samples[k][zone1[0]-2000:zone1[1]-2000]*1e4, 'k',  label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]*1e-4
        plt.plot(range(zone1[0],zone1[1]), W_proj[zone1[0]-2000:zone1[1]-2000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    plt.grid()
    path = "Results/plots_reunion2406/zoom1_"+col+".png"
    plt.savefig(path)
    plt.show()

# ###ZOOM IN ROI 2### PLOT THE W MATRIX ASSOCIATED TO EACH ONE OF THE 3 RM VIEWS 
for k, col in enumerate(ph.columns):
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    plt.plot(range(zone2[0],zone2[1]), res_samples[k][zone2[0]-2000:zone2[1]-2000]*1e4, 'k--',  label=label1)
    plt.plot(range(zone2[0],zone2[1]), sen_samples[k][zone2[0]-2000:zone2[1]-2000]*1e4, 'k',  label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]*1e-4
        plt.plot(range(zone2[0],zone2[1]), W_proj[zone2[0]-2000:zone2[1]-2000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    plt.grid()
    path = "Results/plots_reunion2406/zoom2_"+col+".png"
    plt.savefig(path)
    plt.show()

# ###ZOOM IN ROI 3### PLOT THE W MATRIX ASSOCIATED TO EACH ONE OF THE 3 RM VIEWS 
for k, col in enumerate(ph.columns):
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    
    plt.plot(range(zone3[0],zone3[1]), res_samples[k][zone3[0]-2000:zone3[1]-2000]*1e4, 'k--',  label=label1)
    plt.plot(range(zone3[0],zone3[1]), sen_samples[k][zone3[0]-2000:zone3[1]-2000]*1e4, 'k',  label=label2)
    
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]*1e-4
        plt.plot(range(zone3[0],zone3[1]), W_proj[zone3[0]-2000:zone3[1]-2000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    plt.grid()
    path = "Results/plots_reunion2406/zoom3_"+col+".png"
    plt.savefig(path)
    plt.show()

# ###ZOOM IN ROI 3PLUS### PLOT THE W MATRIX ASSOCIATED TO EACH ONE OF THE 3 RM VIEWS 
for k, col in enumerate(ph.columns):
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    
    plt.plot(range(zone4[0],zone4[1]), res_samples[k][zone4[0]-2000:zone4[1]-2000]*1e4,'k--', label=label1)
    plt.plot(range(zone4[0],zone4[1]), sen_samples[k][zone4[0]-2000:zone4[1]-2000]*1e4, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]*1e-4
        plt.plot(range(zone4[0],zone4[1]), W_proj[zone4[0]-2000:zone4[1]-2000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    plt.grid()
    path = "Results/plots_reunion2406/zoom3_"+col+".png"
    plt.show()


#%%
##### NOW WE CHECK IF THE INTERESTING PEAKS ARE CONSISTENT FOR ALL RESISTENT AND SUSCEPTIBLE SAMPLES

zone1 = [5000,6000]

lf = [np.argsort(-np.abs(results[model_name].q_dist.W[1]['mean'])).squeeze()[0, :3],
np.argsort(-np.abs(results[model_name].q_dist.W[1]['mean'])).squeeze()[1, :3],
np.argsort(-np.abs(results[model_name].q_dist.W[1]['mean'])).squeeze()[2, :3]]

ones_samples = [[], [], []]
zeros_samples = [[], [], []]
for k, col in enumerate(ph.columns):
    ones_samples[k]=np.vstack(x[ph[col]==1])
    zeros_samples[k]=np.vstack(x[ph[col]==0])
    
label1="Sample with a 1"

# ###ZOOM IN ROI 1### Check if the peak exists for all the samples

k=2
col = 'Fenotipo no cpesbl'
for sample in range(ones_samples[k].shape[0]):
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+col+" view")
    plt.plot(range(zone1[0],zone1[1]), ones_samples[k][sample][zone1[0]-2000:zone1[1]-2000]*1e4, 'k--',  label=label1)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]*1e-4
        plt.plot(range(zone1[0],zone1[1]), W_proj[zone1[0]-2000:zone1[1]-2000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    plt.grid()
    path = "Results/plots_reunion2406/zoom1_"+col+".png"
    plt.savefig(path)
    plt.show()
    input()
    plt.close()

################# Project to 3D

# ph_val = ph.loc[folds["val"][f]].values

# ph_tr = results[model_name].t[1]['mean'][:-ph_val.shape[0],:]

# for k, col in enumerate(ph.columns):
#     ord_k = np.argsort(-np.abs(results[model_name].q_dist.W[1]['mean'])[k,:])
#     Z_tr = results[model_name].q_dist.Z['mean'][:, ord_k[:3].astype(int)][:-ph_val.shape[0],:]
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title(col+" projected over Z")
#     scatter1 = ax.scatter(Z_tr[:, 0], Z_tr[:, 1], Z_tr[:, 2], c=ph_tr[:, k])
#     legend1 = ax.legend(*scatter1.legend_elements(), loc='best', title="Classes")
#     ax.add_artist(legend1)

#     plt.legend()
#     ax.set_xlabel('Latent feature '+str(int(ord_k[0])))
#     ax.set_ylabel('Latent feature '+str(int(ord_k[1])))
#     ax.set_zlabel('Latent feature '+str(int(ord_k[2])))
#     ax.legend()

#     plt.show()

# Z_tst = results[model_name].q_dist.Z['mean'][:, ord_k[:3].astype(int)][-ph_val.shape[0]:,:]


# for k, col in enumerate(ph.columns):
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title(col+" projected over Z")
#     scatter1 = ax.scatter(Z_tst[:, 0], Z_tst[:, 1], Z_tst[:, 2], c=ph_val[:, k])
#     legend1 = ax.legend(*scatter1.legend_elements(), loc='best', title="Classes")
#     ax.add_artist(legend1)

#     plt.legend()
#     ax.set_xlabel('Latent feature '+str(int(ord_k[0])))
#     ax.set_ylabel('Latent feature '+str(int(ord_k[1])))
#     ax.set_zlabel('Latent feature '+str(int(ord_k[2])))
#     ax.legend()

#     plt.show()






# %%
