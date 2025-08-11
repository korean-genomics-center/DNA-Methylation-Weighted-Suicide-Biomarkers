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
from itertools import combinations, chain

#%%
##########################################
# Panel A
format_path_ba11_result = "./Data/Validation/Brain_Tissue/randomforest_result.CV_3_Split.seed2.fold_{fold}.suicidal_biomarker.Weighting_Count_Suicidal_Event_FillNA1_Squared.fdr_05.methdiff_5.0.count_90perc.train_from_brain.epic_annot.drop_NA_from_GSE88890.tissue_BA11.cg_near0bp.test.tsv"
# Panel B
format_path_ba11_model = "./Data/Validation/Brain_Tissue/randomforest_model.CV_3_Split.seed2.fold_{fold}.suicidal_biomarker.Weighting_Count_Suicidal_Event_FillNA1_Squared.fdr_05.methdiff_5.0.count_90perc.train_from_brain.epic_annot.drop_NA_from_GSE88890.tissue_BA11.cg_near0bp.pickle"
path_ba11_feature = "/BiO/Research/Project4/Project1/StressomicsMethylome/Stressomics_10_fold_CV_DMP/Resources/Validation_data/GSE88890/Filtered/GSE88890_normalisedBetas.tissue_BA11.cg_selection_epic_near0bp.drop_sample_na_over_None.tsv"
format_path_ba11_meta = ".Data/Validation/Brain_Tissue/GSE88890_series_matrix.melted.necessary.tissue_BA11.cg_selection_epic_near0bp.drop_sample_na_over_None.CV_3_Split.seed2.fold_{fold}.train_test.tsv"

##########################################
#%%
# Read EPIC Array Config
## wget https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/methylationepic/InfiniumMethylationEPICv2.0ProductFiles(ZIPFormat).zip
table_epic = pd.read_csv("./Data/EPIC-8v2-0_A1.csv", skiprows = 7)
#%%
# Draw ROC curve
def draw_roc_prc(path_table, col_y_true, col_y_prob, col_y_pred, name_case = None, draw_fig = True, draw_roc_curve = True, draw_prc_curve = True, overlap_figures = False, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "", name_control = "", ax = None, return_fprtpr = False, kwargs_line = {}, kwargs_marker = {}):
    table = pd.read_csv(path_table, sep = '\t')
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
        if return_fprtpr:
            return (fpr, tpr)
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
    if label_name == None:
        label = None
    else:
        label = f"{label_name} (AUC: {format(auc, '.3f')})"
    ax.plot(fpr, tpr, label = label, zorder = 2, **line_param)
    if mark_best_threshold:
        marker_param = dict(marker='X', c = 'k', s = 40)
        marker_param.update(kwargs_marker)
        ax.scatter([fpr[youden_ind]], [tpr[youden_ind]], label = f"Best threshold :\n  TPR : {format(tpr[youden_ind], '.3f')}\n  FPR : {format(fpr[youden_ind], '.3f')}", zorder = 3, **marker_param)
    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if label != None:
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

def get_feature_importance_from_model(path_model, path_feat, path_meta, cols_featname):
    with open(path_model, 'rb') as fr:
        model = pickle.load(fr)
        print(len(model))
        model = model[0]
    
    table_feat = pd.read_csv(path_feat, sep = '\t')
    table_meta = pd.read_csv(path_meta, sep = '\t')
    
    name_features = table_feat.apply(lambda x : '_'.join(list(map(str, x[cols_featname].to_list()))), axis = 1).to_list()
    
    col_meta_pred = "Sample_Group_Bin"
    col_meta_sampleid = "Sample_ID"
    col_meta_dataset = "Sample_Split"
    cols_meta_feat = list()
    
    (table_feat_train, list_pred_train), (table_feat_test, list_pred_test) = arrange_tables_for_training(table_feat, table_meta, col_meta_pred, col_meta_sampleid, col_meta_dataset, cols_meta_feat)
    
    table_feat_train_values = table_feat_train.T.values
    table_feat_test_values = table_feat_test.T.values
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(table_feat_train_values)
    shap_values.feature_names = name_features    
    
    list_shap_values_for_feat = list(map(lambda x : shap_values[:, x, -1].values, range(shap_values.data.shape[1])))
    list_importance = list(map(lambda x : sum(list(map(abs, x )))/shap_values.data.shape[1], list_shap_values_for_feat))

    dict_importance = dict(zip(shap_values.feature_names, list_importance))
    
    return dict_importance

#%%
plt.rcParams['font.family'] = 'Sans-serif'
plt.rcParams['font.size'] = 8.5
fontsize_label = 12

fig = plt.figure(figsize=(4.76, 2.25))
row = 225
col = 476
# fontsize_label = 17
gsfig = gridspec.GridSpec(
    row, col, 
    left=0, right=1, bottom=0,
    top=1, wspace=1, hspace=1)


gs1 = gsfig[0:175, 0:175] # 175 x 175
ax1 = fig.add_subplot(gs1)

# gs2 = gsfig[0:175, 260:435] # 175 x 175
# ax2 = fig.add_subplot(gs2)

# gs3 = gsfig[225:400, 0:175] # 175 x 175
# ax3 = fig.add_subplot(gs3)

# gs4 = gsfig[225:400, 355:476] # 175 x 121
gs4 = gsfig[0:175, 355:476] # 175 x 121
ax4 = fig.add_subplot(gs4)


roc_color = "firebrick"

# Panel A
list_fpr = list()
list_tpr = list()
for fold in range(3):
    fpr, tpr = draw_roc_prc(format_path_ba11_result.format(fold = fold), "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = False, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = None, name_control = "Control", ax = ax1, return_fprtpr = True, kwargs_line = {"alpha" : 0.4, "color" : roc_color, "linewidth" :1 }, kwargs_marker = {})
    list_fpr.append(fpr)
    list_tpr.append(tpr)
    
base_fpr = np.linspace(0, 1, 501)
list_tpr_adj = list()
list_auc = list()
for fpr, tpr in zip(list_fpr, list_tpr):
    tpr_adj = np.interp(base_fpr, fpr, tpr)
    tpr_adj[0] = 0
    list_tpr_adj.append(tpr_adj)
    list_auc.append(auc(fpr, tpr))

mean_tpr_adj = np.array(list_tpr_adj).mean(axis = 0)
std_tpr_adj = np.array(list_tpr_adj).std(axis = 0)

tpr_adj_upper = np.minimum(mean_tpr_adj + std_tpr_adj, 1)
tpr_adj_lower = mean_tpr_adj - std_tpr_adj

auc_mean = auc(base_fpr, mean_tpr_adj)

ax1.plot(base_fpr, mean_tpr_adj, color = roc_color, linewidth = 1.7, label = f"Mean ROC\n (Mean AUC: {auc_mean:.3f})")
ax1.fill_between(base_fpr, tpr_adj_lower, tpr_adj_upper, color = "gray", alpha = 0.3, label = "±1 Std. dev.")
ax1.plot([0,1], [0,1], c = "gray", linewidth = 1, label = "No Skill (AUC: 0.5)", zorder = 1)
ax1.legend(loc = "lower right", bbox_to_anchor = (0.99, 0.01), fontsize = plt.rcParams["font.size"] - 2, title = "Brain Tissue (BA11)", title_fontproperties = {"size":plt.rcParams["font.size"]-1, "weight" : "bold"})
ax1.set_xticks(np.linspace(0, 1, 6))
ax1.set_yticks(np.linspace(0, 1, 6))
ax1.text(-0.29, 1.01, "A", transform=ax1.transAxes, 
            size=fontsize_label, weight='bold')


# Panel D
list_dict_imp = list(map(lambda fold: get_feature_importance_from_model(
    format_path_ba11_model.format(fold = fold), 
    path_ba11_feature, 
    format_path_ba11_meta.format(fold = fold), 
    ["CG_ID"]), range(3)))

import numpy as np

dict_mean_imp = dict()
list_features = list(list_dict_imp[0].keys())
dict_mean_imp = dict(zip(list_features, list(map(lambda x : np.mean(list(map(lambda y : y[x], list_dict_imp))), list_features))))

list_features_sorted_by_mean_imp = sorted(dict_mean_imp.keys(), key = dict_mean_imp.__getitem__)

table_feature_to_importance = pd.DataFrame(columns = ["CG_ID", "Fold", "Importance", "chr", "pos"])
ind = 0
for feat in list_features_sorted_by_mean_imp:
    dict_epic = table_epic[table_epic["Name"] == feat].iloc[0, :].to_dict()
    chrname = dict_epic["CHR"]
    position = dict_epic["MAPINFO"]
    for fold in range(3):
        table_feature_to_importance.loc[ind, :] = [feat, fold, list_dict_imp[fold][feat], chrname, position]
        ind += 1

def get_list_mean_imp(cg_ids):
    return list(map(dict_mean_imp.__getitem__, cg_ids))

cg_emphasize = ["cg16229848", "cg20178603", "cg05942022"]
dict_color = {
    True : "firebrick",
    False : (179/255,179/255,179/255)
}

table_feature_to_importance_top10 = table_feature_to_importance[table_feature_to_importance["CG_ID"].isin(list_features_sorted_by_mean_imp[-10:])].sort_values(by = "CG_ID", key = get_list_mean_imp, ascending = False)
table_feature_to_importance_top10["CG_ID_repr"] = table_feature_to_importance_top10.apply(lambda x : f"{x['CG_ID']} ({x['chr']}:{int(x['pos'])})", axis = 1)
table_feature_to_importance_top10["Color"] = table_feature_to_importance_top10["CG_ID"].apply(lambda x : dict_color[x in cg_emphasize])
dict_cgrepr_to_color = dict(zip(table_feature_to_importance_top10["CG_ID_repr"], table_feature_to_importance_top10["Color"]))

sns.boxplot(data = table_feature_to_importance_top10, x = "Importance", y = "CG_ID_repr", palette = dict_cgrepr_to_color, ax = ax4, linewidth = 0.7)
ax4.set_ylabel("")
ax4.set_xlabel("Feature Importance (SHAP)")
ax4.spines[['right', 'top']].set_visible(False)

for yticklabel in ax4.get_yticklabels():
    yticklabel.set_fontsize(plt.rcParams["font.size"]-1.5)
    color = dict_cgrepr_to_color[yticklabel.get_text()]
    if color == dict_color[True]:
        yticklabel.set_color(color)
        yticklabel.set_fontweight("bold")
ax4.text(-1.205, 1.01, "B", transform=ax4.transAxes, 
            size=fontsize_label, weight='bold')

gsfig.tight_layout(fig)
plt.savefig("./Figures/Figure_4.pdf", dpi = 650, bbox_inches='tight')
plt.savefig("./Figures/Figure_4.png", dpi = 650, bbox_inches='tight')
plt.show()
plt.close()
# %%
