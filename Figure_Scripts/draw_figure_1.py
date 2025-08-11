#%%
import os
from pathlib import Path
os.chdir(str(Path(__file__).parents[1]))
import math
import scipy.stats as stats
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import gridspec
from statannotations.Annotator import Annotator
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import seaborn as sns
import math
import shap
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
import matplotlib as mpl
import random, pickle
from itertools import combinations


#%%
##################################
# Panel A
path_dmp_all = "./Data/DMS_Result.Control_Normal.Case_Suicide_attempt.set_Train.Cov_Sex_Age.customDMP.weight_Count_Suicidal_Event_FillNA1_Squared.tsv"
path_bootstrap = "./Data/significant_marker_list.Weighting_Count_Suicidal_Event_FillNA1_Squared.fdr_05.methdiff_5.count_90perc.txt"

# Panel B
path_model_result_all = "./Data/ML/randomforest_result.suicidal_biomarker.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.test.tsv"

# Panel C
path_feat_train = "./Data/Raw_CpG_Values.suicidal_biomarker.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.tsv"
path_meta = "./Data/total_samples.non_lith.sample_filtered.pca_filtered.extra_control_filtered.noQueOutlier.SA_N.Normal.seed0.kmeans.SA.UH_ADD_vs_SNU.train_test.binarized.tsv"
path_model = "./Data/ML/randomforest_model.suicidal_biomarker.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.pickle"

name_of_feat = {
    "chr3:52773754" : "$\it{NEK4}$~$\it{ITIH1}$ (Inter.)",
    "chr12:106022437" : "Distal Enhancer",
    "chr12:107775813" : "$\it{ASCL4}$ (Exon)",
    "chr2:123061689" : "Distal Enhancer",
    "chr19:1039445" : "$\it{ABCA7}$ (Promoter)",
    "chr1:42936920" : "$\it{SLC2A1}$ (Intron)",
    "chr12:1922339" : "$\it{CACNA2D4}$ (Promoter)",
    "chr17:42023858" : "$\it{DNAJC7}$ (Promoter) / $\it{NKIRAS2}$ (Exon)",
    "chr1:119713529" : "$\it{PHGDH}$ (Intron)",
    "chr1:33180370" : "$\it{TRIM62}$ (Promoter)"
}

# Panel D
path_model_result_top4 = "./Data/ML/randomforest_result.suicidal_biomarker.Weight_N_Attempts_FillNA1_Squared.fdr_05.methdiff_5.count_90perc.High_Imp_Top_4.test.tsv"

#%%
# Organize DMP result
table_dmp = pd.read_csv(path_dmp_all, sep = '\t')
table_bootstrap = pd.read_csv(path_bootstrap, sep = '\t')
table_dmp["PosName"] = table_dmp.apply(lambda x : f"{x['chr']}:{x['start']}", axis = 1)
table_bootstrap["PosName"] = table_bootstrap.apply(lambda x : f"{x['chr']}:{x['start']}", axis = 1)

table_dmp["Bootstrap_90"] = False
table_dmp.loc[table_dmp["PosName"].isin(table_bootstrap["PosName"]), "Bootstrap_90"] = True

dict_bootstrap_dir = dict(zip(table_bootstrap["PosName"], table_bootstrap["direction"]))
table_dmp["Bootstrap_direction"] = table_dmp["PosName"].apply(lambda x : dict_bootstrap_dir.get(x))
table_dmp["minus_log10_fdr"] = table_dmp["qvalue"].apply(lambda x : -math.log10(x))
#%%
# Draw ROC curve
def draw_roc_prc(table, col_y_true, col_y_prob, col_y_pred, name_case = None, draw_fig = True, draw_roc_curve = True, draw_prc_curve = True, overlap_figures = False, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "", name_control = "", ax = None, kwargs_line = {}, kwargs_marker = {}):
    # table = pd.read_csv(path_table, sep = '\t')
    y_true = table[col_y_true].to_list()
    count_true = Counter(y_true)
    # min_count = min(count_true.values())
    y_pred = table[col_y_pred].to_list()
    y_prob = table[col_y_prob].to_list()
    fpr, tpr, thresholds = get_roc(y_true, y_prob, name_case)
    if choose_balanced:
        optimal_ind = get_balanced_index(tpr, fpr)
    elif choose_best_acc:
        optimal_ind = get_maximum_acc(y_true, y_prob, thresholds, name_case, name_control)
    else:
        optimal_ind = get_youden_index(tpr, fpr)
    best_threshold = thresholds[optimal_ind]
    y_pred_best = list(map(lambda x : name_case if x >= best_threshold else name_control, y_prob))
    precision, recall, thresholds_prc = get_prc(y_true, y_prob, name_case)
    auroc = auc(fpr, tpr)
    if draw_fig and draw_roc_curve:
        draw_roc(fpr, tpr, mark_best_threshold, optimal_ind, label_name, auroc, overlap_figures, draw_baseline, ax, kwargs_line, kwargs_marker)
    auprc = auc(recall, precision)
    if draw_fig and draw_prc_curve:
        optimal_ind_prc = list(thresholds_prc).index(best_threshold)
        draw_prc(recall, precision, count_true[name_case]/len(y_true), mark_best_threshold, optimal_ind_prc, label_name, auprc, overlap_figures, draw_baseline, ax)
    return auroc
    
def get_youden_index(tpr, fpr):
    return np.argmax(np.array(tpr)-np.array(fpr))

def get_balanced_index(tpr, fpr):
    return np.argmin(list(map(abs,np.array(tpr)-(1-np.array(fpr)))))
    
def get_maximum_acc(y_true, y_prob, thresholds, name_case, name_control):
    list_acc = list()
    for threshold in thresholds:
        y_pred = list(map(lambda x : name_control if x < threshold else name_case, y_prob))
        acc = accuracy_score(y_true, y_pred)
        list_acc.append(acc)
    return np.argmax(list_acc)
    
def get_roc(y_true, y_pred, name_case):
    fpr, tpr, thresholds = roc_curve(y_true = y_true, y_score = y_pred, pos_label = name_case)
    return fpr, tpr, thresholds

def get_prc(y_true, y_pred, name_case):
    precision, recall, thresholds = precision_recall_curve(y_true = y_true, probas_pred = y_pred, pos_label = name_case)
    return precision, recall, thresholds

def get_precision_recall_curve(y_true, probas_pred, name_pos):
    cutoffs = [min(probas_pred)-1] + sorted(probas_pred)
    print(cutoffs)
    list_precision = list()
    list_recall = list()
    for cutoff in cutoffs:
        prediction_by_cutoff = list(map(lambda x : name_pos if x > cutoff else None, probas_pred))
        table_tmp = pd.DataFrame({"Answer":y_true, "Prediction":prediction_by_cutoff})
        table_pos = table_tmp[table_tmp["Prediction"] == name_pos]
        table_neg = table_tmp[table_tmp["Prediction"] != name_pos]
        
        num_tp = table_pos[table_pos["Answer"] == name_pos].shape[0]
        num_fp = table_pos[table_pos["Answer"] != name_pos].shape[0]
        num_tn = table_neg[table_neg["Answer"] != name_pos].shape[0]
        num_fn = table_neg[table_neg["Answer"] == name_pos].shape[0]
        tot_num = table_tmp.shape[0]
        tpp = num_tp / tot_num * 100
        fpp = num_fp / tot_num * 100
        tnp = num_tn / tot_num * 100
        fnp = num_fn / tot_num * 100
        
        if tpp+fpp == 0:
            precision = 1
        else:
            precision = tpp / (tpp + fpp)
        print(tpp+fnp)
        recall = tpp / (tpp + fnp)
        list_precision.append(precision)
        list_recall.append(recall)
    return np.array(list_precision), np.array(recall), np.array(cutoffs)

def draw_roc(fpr, tpr, mark_best_threshold, youden_ind, label_name, auc, overlap_figures, draw_baseline, ax, kwargs_line = {}, kwargs_marker = {}):
    mpl.rcParams["axes.axisbelow"] = True
    if type(ax) == type(None):
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
    if draw_baseline:
        ax.plot([0,1], [0,1], c = "gray", linewidth = 1, label = "No Skill (AUC: 0.5)", zorder = 1)
    line_param = {"linewidth" : 3}
    line_param.update(kwargs_line)
    ax.plot(fpr, tpr, label = f"{label_name} (AUC: {format(auc, '.3f')})", zorder = 2, **line_param)
    if mark_best_threshold:
        marker_param = dict(marker='X', c = 'k', s = 40)
        marker_param.update(kwargs_marker)
        ax.scatter([fpr[youden_ind]], [tpr[youden_ind]], label = f"Optimal threshold :\n  TPR : {format(tpr[youden_ind], '.3f')}\n  FPR : {format(fpr[youden_ind], '.3f')}", zorder = 3, **marker_param)
    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc = "lower right", bbox_to_anchor = (0.99, 0.01), fontsize = plt.rcParams["font.size"] - 1)
    ax.grid(linestyle='--', linewidth = 0.5, color = "gray")
    if not overlap_figures:
        print(1)
        plt.show()

def draw_prc(recall, precision, random_value, mark_best_threshold, youden_ind, label_name, auc, overlap_figures, draw_baseline, ax):
    mpl.rcParams["axes.axisbelow"] = True
    if type(ax) == type(None):
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
    if draw_baseline:
        plt.hlines(random_value, 0, 1, colors = "gray", linewidth = 1.5, label = f"No Skill (AUC: {round(random_value, 3)})", zorder = 1)
    plt.plot(recall, precision, c = 'r', linewidth = 3, label = f"{label_name} (AUC: {round(auc, 3)})", zorder = 2)
    if mark_best_threshold:
        plt.scatter([recall[youden_ind]], [precision[youden_ind]], marker='X', c = 'k', s = 40, label = f"Best threshold :\n  Precision : {round(precision[youden_ind], 5)}\n  Recall : {round(recall[youden_ind], 5)}", zorder = 3)
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(linestyle='--')
    if not overlap_figures:
        plt.show()

#%%
# Get SHAP values
with open(path_model, 'rb') as fr:
    rfmodel = pickle.load(fr)[0]
table_feat = pd.read_csv(path_feat_train, sep = '\t')
table_meta = pd.read_csv(path_meta, sep = '\t')
cols_featname = ["chr", "start", "end"]
name_features = table_feat.apply(lambda x : '_'.join(list(map(str, x[cols_featname].to_list()))), axis = 1).to_list()

def arrange_tables_for_training(table_values, table_meta, col_meta_pred, col_meta_sampleid, col_meta_dataset, cols_meta_feat):    
    table_meta_train = table_meta[table_meta[col_meta_dataset] == "Train"]
    table_meta_test = table_meta[table_meta[col_meta_dataset] == "Test"]
    table_meta_eval = table_meta[table_meta[col_meta_dataset] == "Eval"]
    assert table_meta_train.shape[0] > 0, "No Train Dataset"
    assert table_meta_test.shape[0] > 0, "No Test Dataset"
    list_train_samples = table_meta_train[col_meta_sampleid].to_list()
    list_test_samples = table_meta_test[col_meta_sampleid].to_list()
    table_values = table_values[list_train_samples + list_test_samples]
    print(table_values.shape)
    if len(cols_meta_feat) > 0:
        for ind, col_feat in enumerate(cols_meta_feat):
            feat_meta = table_meta[col_feat].to_list()
            print(len(feat_meta))
            table_values.loc[table_values.shape[0]+ind, :] = feat_meta
    print(table_values.shape)
    
    table_feat_train = table_values.loc[:, list_train_samples]
    table_feat_test = table_values.loc[:, list_test_samples]
    
    list_pred_train = table_meta_train[col_meta_pred].to_list()
    list_pred_test = table_meta_test[col_meta_pred].to_list()
    
    return (table_feat_train, list_pred_train), (table_feat_test, list_pred_test)

col_meta_pred = "Sample_Group_Bin"
col_meta_sampleid = "Sample_ID"
col_meta_dataset = "Sample_Split"
cols_meta_feat = list()
(table_feat_train, list_pred_train), (table_feat_test, list_pred_test) = arrange_tables_for_training(table_feat, table_meta, col_meta_pred, col_meta_sampleid, col_meta_dataset, cols_meta_feat)

table_feat_train_values = table_feat_train.T.values
table_feat_test_values = table_feat_test.T.values

name_features_readable = list(map(lambda x : f"{x.split('_')[0]}:{x.split('_')[1]}", name_features))
rfexplainer = shap.TreeExplainer(rfmodel)
rfshap_values = rfexplainer(table_feat_train_values)
rfshap_values.feature_names = name_features_readable

list_rf_shap_values_for_feat = list(map(lambda x : rfshap_values[:, x, -1].values, range(len(table_feat_train_values[0]))))
list_rf_importance = list(map(lambda x : sum(list(map(abs, x )))/len(table_feat_train_values), list_rf_shap_values_for_feat))
dict_importance_rf = dict(zip(name_features_readable, list_rf_importance))


#%%
plt.rcParams['font.family'] = 'Sans-serif'
plt.rcParams['font.size'] = 8
fontsize_label = 10

fig = plt.figure(figsize=(4.76, 4))
row = 400
col = 476
# fontsize_label = 17
gsfig = gridspec.GridSpec(
    row, col, 
    left=0, right=1, bottom=0,
    top=1, wspace=1, hspace=1)


gs1 = gsfig[0:175, 0:170] # 175 x 170
ax1 = fig.add_subplot(gs1)

gs2 = gsfig[0:175, 301:476] # 175 x 175
ax2 = fig.add_subplot(gs2)

gs3 = gsfig[235:400, 30:150] # 165 x 130
ax3 = fig.add_subplot(gs3)

gs4 = gsfig[225:400, 301:476] # 175 x 175
ax4 = fig.add_subplot(gs4)


# Panel A
sns.scatterplot(data = table_dmp[table_dmp["Bootstrap_90"] == True],
                y = "minus_log10_fdr",
                x = "wmeth.diff",
                color = 'r',
                s = 8,
                marker = 'X',
                zorder = 10,
                ax = ax1)
sns.scatterplot(data = table_dmp[table_dmp["Bootstrap_90"] == False],
                y = "minus_log10_fdr",
                x = "wmeth.diff",
                color = 'grey',
                s = 3,
                marker = '.',
                zorder = 1,
                ax = ax1)
ax1.axvline(5, linestyle = '--', color = 'gray', linewidth = 0.8)
ax1.axvline(-5, linestyle = '--', color = 'gray', linewidth = 0.8)
ax1.axhline(-math.log10(0.05), linestyle = '--', color = "gray", linewidth = 0.8)
ax1.set_xlim([-35, 35])
ax1.set_xticks(range(-30, 31, 10))

ax1_leg = ax1.legend(labels = ["Yes", "No"], title = "Replicated\n(≥90%)", bbox_to_anchor = (1.01, 0.5), loc = "center left", title_fontsize = plt.rcParams["font.size"]-1, frameon=True)
ax1_leg.legendHandles[0]._sizes = [15]
ax1_leg.legendHandles[1]._sizes = [10]
ax1.set_xlabel("Weighted Mean Methylation Difference (%)")
ax1.set_ylabel("-log10(FDR)")
ax1.text(-0.3, 0.97, "A", transform=ax1.transAxes, 
            size=fontsize_label, weight='bold')

# Panel B
draw_roc_prc(path_model_result_all, "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = True, choose_balanced = False, choose_best_acc = False, label_name = "365 Biomarker set\n", name_control = "Control", ax = ax2, kwargs_line = {"linewidth" : 1.7, "color" : "r"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax2.set_xticks(np.linspace(0, 1, 6))
ax2.text(-0.29, 0.97, "B", transform=ax2.transAxes, 
            size=fontsize_label, weight='bold')

# Panel C
features_top10_imp = list(reversed(sorted(dict_importance_rf.keys(), key = dict_importance_rf.__getitem__, reverse = True)[:10]))
impvalues_top10_imp = list(map(dict_importance_rf.__getitem__, features_top10_imp))
ax3.barh(y = features_top10_imp, width = impvalues_top10_imp, color = "gray")
for ind, featname in enumerate(features_top10_imp):
    feat_imp = dict_importance_rf[featname]
    label_featname = name_of_feat.get(featname, "")
    ax3.annotate(label_featname, (feat_imp+0.0005, ind), va = "center", ha = "left", fontsize = plt.rcParams["font.size"]-1.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.tick_params(axis = "y", labelsize = plt.rcParams["font.size"]-1)
ax3.set_ylim(-0.5, 9.5)
ax3.set_xlabel("Feature Importance (SHAP)")
ax3.text(-0.3 * 200/88, 0.97 * 175/165, "C", transform=ax3.transAxes, 
            size=fontsize_label, weight='bold')

# Panel D
draw_roc_prc(path_model_result_top4, "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = True, choose_balanced = False, choose_best_acc = False, label_name = "Top 4 Important set\n", name_control = "Control", ax = ax4, kwargs_line = {"linewidth" : 1.7, "color" : "r"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax4.set_xticks(np.linspace(0, 1, 6))
ax4.text(-0.29, 0.97, "D", transform=ax4.transAxes, 
            size=fontsize_label, weight='bold')


gsfig.tight_layout(fig)
plt.savefig("./Figures/Figure_1.pdf", dpi = 650, bbox_inches='tight')
plt.savefig("./Figures/Figure_1.png", dpi = 650, bbox_inches='tight')
plt.show()
plt.close()
# %%
