#%%
import os, math, random, pickle
from collections import Counter
from itertools import combinations
from pathlib import Path
os.chdir(str(Path(__file__).parents[1]))

import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import gridspec
from collections import Counter
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score

#%%
path_dmp_noweight = "./Data/Supple/Supple_Fig2/Methylation_DMP_Extract.Control_Normal.Case_Suicide_attempt.sample_filtered.noQueOutlier.SA_N.SA_UH_ADD_vs_SNU.seed0.kmeans.set_Train.Case_UH_ADD.Cov_Sex_Age.customDMP.weight_Empty_weight.tsv"
path_cpgs_noweight_randomsampled = ""
path_roc_noweight = "./Data/Supple/Supple_Fig2/randomforest_result.suicidal_biomarker.Weighting_Empty_weight.fdr_05.wmethdiff_5.test.tsv"
path_roc_noweight_randomsampled = ""

path_dmp_simple = "./Data/Supple/Supple_Fig2/Methylation_DMP_Extract.Control_Normal.Case_Suicide_attempt.sample_filtered.noQueOutlier.SA_N.SA_UH_ADD_vs_SNU.seed0.kmeans.set_Train.Case_UH_ADD.Cov_Sex_Age.customDMP.weight_Count_Suicidal_Event_FillNA1.tsv"
path_cpgs_simple_randomsampled = "./Data/Supple/Supple_Fig2/significant_marker_list.weight_count_suicidal_event.fdr_05.methdiff_5.count_90perc.txt"
path_roc_simple = "./Data/Supple/Supple_Fig2/randomforest_result.suicidal_biomarker.Weighting_Count_Suicidal_Event_FillNA1.fdr_05.wmethdiff_5.test.tsv"
path_roc_simple_randomsampled = "./Data/Supple/Supple_Fig2/randomforest_result.suicidal_biomarker.weight_count_suicidal_event.fdr_05.methdiff_5.count_90perc.test.tsv"

path_dmp_squared = "./Data/Supple/Supple_Fig2/Methylation_DMP_Extract.Control_Normal.Case_Suicide_attempt.sample_filtered.noQueOutlier.SA_N.SA_UH_ADD_vs_SNU.seed0.kmeans.set_Train.Case_UH_ADD.Cov_Sex_Age.customDMP.weight_Count_Suicidal_Event_FillNA1_Squared.tsv"
path_cpgs_squared_randomsampled = "./Data/Supple/Supple_Fig2/significant_marker_list.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.txt"
path_roc_squared = "./Data/Supple/Supple_Fig2/randomforest_result.suicidal_biomarker.Weighting_Count_Suicidal_Event_FillNA1_Squared.fdr_05.wmethdiff_5.test.tsv"
path_roc_squared_randomsampled = "./Data/Supple/Supple_Fig2/randomforest_result.suicidal_biomarker.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.test.tsv"

path_save_prefix = "./Figures/Supplementary_Figure2"

#%%
table_dmp_noweight = pd.read_csv(path_dmp_noweight, sep = '\t')
table_dmp_simple = pd.read_csv(path_dmp_simple, sep = '\t')
table_dmp_squared = pd.read_csv(path_dmp_squared, sep = '\t')

table_roc_noweight = pd.read_csv(path_roc_noweight, sep = '\t')
table_roc_simple = pd.read_csv(path_roc_simple, sep = '\t')
table_roc_squared = pd.read_csv(path_roc_squared, sep = '\t')

table_roc_noweight_randomsampled = pd.DataFrame()
table_roc_simple_randomsampled = pd.read_csv(path_roc_simple_randomsampled, sep = '\t')
table_roc_squared_randomsampled = pd.read_csv(path_roc_squared_randomsampled, sep = '\t')
#%%
# Draw ROC curve
def draw_roc_prc(table, col_y_true, col_y_prob, col_y_pred, name_case = None, draw_fig = True, draw_roc_curve = True, draw_prc_curve = True, overlap_figures = False, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "", name_control = "", ax = None, kwargs_line = {}, kwargs_marker = {}):
    if table.shape[0] == 0:return
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
# Draw Volcano
def draw_volcano(table_dmp, overlap_figures, ax, cpgs_emp = '', kwargs_marker_emp = {}, draw_legend = False, kwargs_line = {}, kwargs_marker = {}):
    color_palette = {
        True:"firebrick",
        False:"gray"
    }
    
    list_cpgs_emp = list()
    if os.path.exists(cpgs_emp):
        table_cpgs_emp = pd.read_csv(cpgs_emp, sep = '\t')
        table_cpgs_emp["cpgname"] = table_cpgs_emp.apply(lambda row: f"{row['chr']}:{row['start']}:{row['end']}", axis = 1)
        list_cpgs_emp = table_cpgs_emp["cpgname"].to_list()        
    
    table_dmp_draw = table_dmp.copy()
    
    table_dmp_draw["is_sig"] = table_dmp_draw.apply(lambda row: True if abs(row["wmeth.diff"]) > 5 and row["qvalue"] < 0.05 else False, axis = 1)
    table_dmp_draw["minus_log10_fdr"] = table_dmp_draw["qvalue"].apply(lambda val: -math.log10(val))
    
    table_dmp_draw["cpgname"] = table_dmp_draw.apply(lambda row: f"{row['chr']}:{row['start']}:{row['end']}", axis = 1)
    table_dmp_draw["is_emp"] = table_dmp_draw["cpgname"].apply(lambda val: val in list_cpgs_emp)
    
    mpl.rcParams["axes.axisbelow"] = True
    if type(ax) == type(None):
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        
    line_param = {"linewidth" : 0.8, "linestyle" : "--", "color" : "gray"}
    line_param.update(kwargs_line)
    
    table_dmp_draw_emp = table_dmp_draw[table_dmp_draw["is_emp"] == True]
    table_dmp_draw_other = table_dmp_draw[table_dmp_draw["is_emp"] != True]
    
    kwargs_marker_emp.update(kwargs_marker)
    sns.scatterplot(data = table_dmp_draw_emp,
                    y = "minus_log10_fdr",
                    x = "wmeth.diff",
                    zorder = 15,
                    ax = ax,
                    **kwargs_marker_emp)
    sns.scatterplot(data = table_dmp_draw_other[table_dmp_draw_other["is_sig"] == True],
                    y = "minus_log10_fdr",
                    x = "wmeth.diff",
                    color = 'royalblue',
                    s = 8,
                    marker = 'X',
                    zorder = 10,
                    ax = ax,
                    **kwargs_marker)
    sns.scatterplot(data = table_dmp_draw_other[table_dmp_draw_other["is_sig"] == False],
                    y = "minus_log10_fdr",
                    x = "wmeth.diff",
                    color = 'grey',
                    s = 3,
                    marker = '.',
                    zorder = 1,
                    ax = ax,
                    **kwargs_marker)
    
    ax.axvline(5, **line_param)
    ax.axvline(-5, **line_param)
    ax.axhline(-math.log10(0.05), **line_param)
    
    ax.set_xlim([-35, 35])
    ax.set_xticks([-30, -15, -5, 5, 15, 30])
    
    ax.set_xlabel("Weighted Mean Methylation Difference (%)")
    ax.set_ylabel("-log10(FDR)")
    
    ax.annotate(
        f"{sum(table_dmp_draw['is_sig']):,}",
        (0.96, 0.93),
        xycoords = "axes fraction",
        ha = "right",
        va = "top",
        fontweight = "bold",
        fontsize = plt.rcParams["font.size"] + 1,
        color = "royalblue",
        zorder = 20
    )
    ax.annotate(
        f"{len(list_cpgs_emp):,}",
        (0.96, 0.85),
        xycoords = "axes fraction",
        ha = "right",
        va = "top",
        fontweight = "bold",
        fontsize = plt.rcParams["font.size"] + 1,
        color = "r",
        zorder = 20
    )

    if draw_legend:
        ax_leg = ax.legend(ncol = 3, labels = ["Significant (Random Sampled)", "Significant", "Non-Significant"], bbox_to_anchor = (-0.2, -0.25), loc = "upper left", title_fontsize = plt.rcParams["font.size"]+3, frameon=True)
        ax_leg.legendHandles[0]._sizes = [20]
        ax_leg.legendHandles[1]._sizes = [20]
        ax_leg.legendHandles[2]._sizes = [13]
    
    if not overlap_figures:
        plt.show()

#%%
plt.rcParams['font.family'] = 'Sans-serif'
plt.rcParams['font.size'] = 9
mpl.rcParams["axes.axisbelow"] = True
fontsize_label = 10


xspace = 70
yspace = 55

volcano_x = 140
volcano_y = 175
roc_x = 175
roc_y = 175

row = volcano_y*3+yspace*2
col = volcano_x+roc_x*2+xspace*2
fig = plt.figure(figsize=(col/100, row/100))
gsfig = gridspec.GridSpec(
    row, col, 
    left=0, right=1, bottom=0,
    top=1, wspace=1, hspace=1)


gs11 = gsfig[volcano_y*0+yspace*0:volcano_y*1+yspace*0, 0:volcano_x] # 175 x 170
ax11 = fig.add_subplot(gs11)

gs12 = gsfig[roc_y*0+yspace*0:roc_y*1+yspace*0, volcano_x+xspace:volcano_x+xspace+roc_x] # 175 x 175
ax12 = fig.add_subplot(gs12)

gs13 = gsfig[roc_y*0+yspace*0:roc_y*1+yspace*0, volcano_x+xspace*2+roc_x:volcano_x+xspace*2+roc_x*2] # 175 x 175
ax13 = fig.add_subplot(gs13)

gs21 = gsfig[volcano_y*1+yspace*1:volcano_y*2+yspace*1, 0:volcano_x] # 175 x 170
ax21 = fig.add_subplot(gs21)

gs22 = gsfig[roc_y*1+yspace*1:roc_y*2+yspace*1, volcano_x+xspace:volcano_x+xspace+roc_x] # 175 x 175
ax22 = fig.add_subplot(gs22)

gs23 = gsfig[roc_y*1+yspace*1:roc_y*2+yspace*1, volcano_x+xspace*2+roc_x:volcano_x+xspace*2+roc_x*2] # 175 x 175
ax23 = fig.add_subplot(gs23)


gs31 = gsfig[volcano_y*2+yspace*2:volcano_y*3+yspace*2, 0:volcano_x] # 175 x 170
ax31 = fig.add_subplot(gs31)

gs32 = gsfig[roc_y*2+yspace*2:roc_y*3+yspace*2, volcano_x+xspace:volcano_x+xspace+roc_x] # 175 x 175
ax32 = fig.add_subplot(gs32)

gs33 = gsfig[roc_y*2+yspace*2:roc_y*3+yspace*2, volcano_x+xspace*2+roc_x:volcano_x+xspace*2+roc_x*2] # 175 x 175
ax33 = fig.add_subplot(gs33)

kwargs_marker_emp = dict(
    color = 'r',
    s = 10,
    marker = 'X'
)

draw_volcano(table_dmp_noweight, True, ax11, cpgs_emp = path_cpgs_noweight_randomsampled, kwargs_marker_emp = kwargs_marker_emp)
ax11.text(-0.3, 0.97, "A", transform = ax11.transAxes,
          size = fontsize_label, weight = "bold")
draw_roc_prc(table_roc_noweight, "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "No Weight\n", name_control = "Control", ax = ax12, kwargs_line = {"linewidth" : 1.7, "color" : "royalblue"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax12.set_xticks(np.linspace(0, 1, 6))
ax12.text(-0.29, 0.97, "B", transform=ax12.transAxes, 
            size=fontsize_label, weight='bold')
ax12.set_title("Without Random Sampling")
draw_roc_prc(table_roc_noweight_randomsampled, "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "No Weight\n", name_control = "Control", ax = ax13, kwargs_line = {"linewidth" : 1.7, "color" : "r"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax13.set_xticks(np.linspace(0, 1, 6))
ax13.annotate("No CpG site", (0.5, 0.5), va = "center", ha = "center", fontweight = "bold", fontsize = plt.rcParams["font.size"]+2)
ax13.text(-0.29, 0.97, "C", transform=ax13.transAxes, 
            size=fontsize_label, weight='bold')
ax13.set_title("With Random Sampling")

ax11.text(-0.45, 0.5, "No Weight", rotation = 90, transform = ax11.transAxes, size = plt.rcParams["font.size"]+2, ha = "center", va = "center")

draw_volcano(table_dmp_simple, True, ax21, cpgs_emp = path_cpgs_simple_randomsampled, kwargs_marker_emp = kwargs_marker_emp)
ax21.text(-0.3, 0.97, "D", transform = ax21.transAxes,
          size = fontsize_label, weight = "bold")
draw_roc_prc(table_roc_simple, "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "Simple Weight\n", name_control = "Control", ax = ax22, kwargs_line = {"linewidth" : 1.7, "color" : "royalblue"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax22.set_xticks(np.linspace(0, 1, 6))
ax22.text(-0.29, 0.97, "E", transform=ax22.transAxes, 
            size=fontsize_label, weight='bold')
draw_roc_prc(table_roc_simple_randomsampled, "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "Simple Weight\n", name_control = "Control", ax = ax23, kwargs_line = {"linewidth" : 1.7, "color" : "r"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax23.set_xticks(np.linspace(0, 1, 6))
ax23.text(-0.29, 0.97, "F", transform=ax23.transAxes, 
            size=fontsize_label, weight='bold')

ax21.text(-0.45, 0.5, "Simple Weight", rotation = 90, transform = ax21.transAxes, size = plt.rcParams["font.size"]+2, ha = "center", va = "center")

draw_volcano(table_dmp_squared, True, ax31, draw_legend = True, cpgs_emp = path_cpgs_squared_randomsampled, kwargs_marker_emp = kwargs_marker_emp)
ax31.text(-0.3, 0.97, "G", transform = ax31.transAxes,
          size = fontsize_label, weight = "bold")
draw_roc_prc(table_roc_squared, "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "Squared Weight\n", name_control = "Control", ax = ax32, kwargs_line = {"linewidth" : 1.7, "color" : "royalblue"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax32.set_xticks(np.linspace(0, 1, 6))
ax32.text(-0.29, 0.97, "H", transform=ax32.transAxes, 
            size=fontsize_label, weight='bold')
draw_roc_prc(table_roc_squared_randomsampled, "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "Squared Weight\n", name_control = "Control", ax = ax33, kwargs_line = {"linewidth" : 1.7, "color" : "r"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax33.set_xticks(np.linspace(0, 1, 6))
ax33.text(-0.29, 0.97, "I", transform=ax33.transAxes, 
            size=fontsize_label, weight='bold')

ax31.text(-0.45, 0.5, "Squared Weight", rotation = 90, transform = ax31.transAxes, size = plt.rcParams["font.size"]+2, ha = "center", va = "center")

plt.savefig(f"{path_save_prefix}.png", dpi = 650, bbox_inches = "tight")
plt.savefig(f"{path_save_prefix}.pdf", dpi = 650, bbox_inches = "tight")
plt.show()
plt.close()
# %%
