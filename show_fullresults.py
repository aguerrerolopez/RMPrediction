import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

######################### PARAMETERS SELECTION ########################################
# Position where YOU introduced the RM view to the model during training
phen_pos = 1

######################### LOAD DATA ########################################
model_path = "./Results/mediana_10fold_rbf/HGM_full_linear_15views_PRUEBALOCA_prun0.1.pkl"
folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_pruebaloca.pkl"
data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"

with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)
with open(model_path, 'rb') as pkl:
    results = pickle.load(pkl)

##### PREDICT CARB AND ESBL
old_fen = gm_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
ph = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
x = gm_data["maldi"].loc[ph.index]
y = gm_data['binary_ab'].loc[ph.index]

###### PREDICT OTHER AB
# x = gm_data["maldi"]
# y = gm_data['binary_ab']


auc_by_ab = np.zeros((10,9))
auc_by_phen = np.zeros((10,3))


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
    y_tr, y_val = y.loc[folds["train"][f]], y.loc[folds["val"][f]]
    ##### PREDICT CARB AND ESBL
    ph_val = ph.loc[folds["val"][f]].values
    y_val = y_val.values
    ###### PREDICT OTHER AB
    # cols = ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA', 'CIPROFLOXACINO', 'COLISTINA']
    # y_val = y_val[cols].values

    model_name = "model_fold" + str(f)
    for h in range(9):
        ##### PREDICT CARB AND ESBL
        y_true = y_val[:, h]
        y_pred = results[model_name].t[2+h]['mean'][-y_val.shape[0]:, 0]
        ###### PREDICT OTHER AB
        # y_pred = results[model_name].t[1+h]['mean'][-y_val.shape[0]:, 0]
        # auc_by_ab[f, h] = roc_auc_score(y_true, y_pred)
        # if h>5:
        #     break

    ### PRED CARB BLEE
    y_true = ph_val
    y_pred = results[model_name].t[1]['mean'][-y_val.shape[0]:, :]
    auc_by_phen[f,0] = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    auc_by_phen[f,1] = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    auc_by_phen[f,2] = roc_auc_score(y_true[:, 2], y_pred[:, 2])
   
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

####################################### PLOT SOME FEATURES IN DETAIL ###################################3
X=results[model_name].X[0]['X']
W_maldi=X.T@results[model_name].q_dist.W[0]['mean']
# SELECT HERE WHICH FEATURES DO YOU WANT TO PLOT IN DETAIL!!!!!
lf = [23, 1, 21]
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



# PLOT THE W MATRIX ASSOCIATED TO EACH ONE OF THE 3 RM VIEWS
X=results[model_name].X[0]['X']
W_maldi=X.T@results[model_name].q_dist.W[0]['mean']
lf = [lf, lf, lf]
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
    path = "Results/plots_reunion2406/privadas_"+col+".png"
    # plt.savefig(path)
    plt.show()


# ###ZOOM IN ROI 1### PLOT THE W MATRIX ASSOCIATED TO EACH ONE OF THE 3 RM VIEWS 
for k, col in enumerate(ph.columns):
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.plot(range(2000,2500), res_samples[k][0:500], '--', alpha=0.2, label=label1)
    plt.plot(range(2000,2500), sen_samples[k][0:500], alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(2000,2500), W_proj[0:500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
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
    plt.plot(range(7000,7500), res_samples[k][5000:5500], '--', alpha=0.2, label=label1)
    plt.plot(range(7000,7500), sen_samples[k][5000:5500], alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(7000,7500), W_proj[5000:5500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
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
    plt.plot(range(9500,10000), res_samples[k][7500:8000], '--', alpha=0.2, label=label1)
    plt.plot(range(9500,10000), sen_samples[k][7500:8000], alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(9500,10000), W_proj[7500:8000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
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
    plt.plot(range(9900,10000), res_samples[k][7900:8000], '--', alpha=0.2, label=label1)
    plt.plot(range(9900,10000), sen_samples[k][7900:8000], alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(9900,10000), W_proj[7900:8000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    path = "Results/plots_reunion2406/zoom3_"+col+".png"
    plt.show()


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

