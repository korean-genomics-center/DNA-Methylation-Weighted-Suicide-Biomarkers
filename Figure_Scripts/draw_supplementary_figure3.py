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
path_train_roc_format = "./Data/Supple/Supple_Fig3/randomforest_result.suicidal_biomarker.Weight_N_Attempts_FillNA1_Squared.fdr_05.methdiff_5.count_90perc.Train_High_Imp_Top_{ntop}.train.tsv"
path_test_roc_format = "./Data/Supple/Supple_Fig3/randomforest_result.suicidal_biomarker.Weight_N_Attempts_FillNA1_Squared.fdr_05.methdiff_5.count_90perc.Train_High_Imp_Top_{ntop}.test.tsv"

path_save_prefix = "./Figures/Supplementary_Figure3"

n_tottop = 50

auroc_fullmodel = 0.760
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


dict_train_ranking_to_auroc = dict()
dict_test_ranking_to_auroc = dict()

for ntop in range(1, n_tottop+1):
    path_train_roc = path_train_roc_format.format(ntop = ntop)
    path_test_roc = path_test_roc_format.format(ntop = ntop)
    
    table_train_roc = pd.read_csv(path_train_roc, sep = '\t')
    table_test_roc = pd.read_csv(path_test_roc, sep = '\t')
    
    fpr_train, tpr_train, _ = get_roc(table_train_roc["Answer"], table_train_roc["Proba_Case"], "Case")
    fpr_test, tpr_test, _ = get_roc(table_test_roc["Answer"], table_test_roc["Proba_Case"], "Case")
    auroc_train = auc(fpr_train, tpr_train)
    auroc_test = auc(fpr_test, tpr_test)
    
    dict_train_ranking_to_auroc[ntop] = auroc_train
    dict_test_ranking_to_auroc[ntop] = auroc_test
#%%

def draw_cumulative_auroc_lineplot(dict_ranking_to_auroc, n_tottop, ax, mark_level = None, mark_name = None, ytick_scale = 0.05):
    auroc_top = 0
    maxima_ntop = 0
    for ntop, auroc in dict_ranking_to_auroc.items():
        if auroc > auroc_top:
            maxima_ntop = ntop
            auroc_top = auroc
    
    ax.plot(dict_ranking_to_auroc.keys(), dict_ranking_to_auroc.values(), linewidth = 1.25, color = "k", zorder = 3)
    
    
    list_auroc_values = list(dict_ranking_to_auroc.values())
    
    ax.axvline(maxima_ntop, linewidth = 1, color = "r", label = f"Highest performance among\nCombinations of CpGs\nordered by importance\n(AUROC: {auroc_top:.3f})")
    
    if mark_level != None:
        ax.axhline(mark_level, linewidth = 1, color = 'k', linestyle = "--", label = mark_name)
        list_auroc_values.append(mark_level)
    
    min_auroc = min(list_auroc_values)
    max_auroc = max(list_auroc_values)
    
    ylim_bottom = math.floor(min_auroc * 0.999 / ytick_scale) * ytick_scale
    ylim_top = math.ceil(max_auroc * 1.001 / ytick_scale) * ytick_scale
    
    ax.set_xticks([1] + list(range(5, 51, 5)), fontsize=  plt.rcParams["font.size"]-1)
    ax.set_yticks(np.arange(ylim_bottom, ylim_top, ytick_scale))
    ax.grid(zorder = 1, linewidth = 0.75, color = "lightgray")
    ax.set_xlabel("Number of included CpGs according to\nthe order of feature importance")
    ax.set_ylabel("AUROC")
    ax.legend(loc = "upper center", bbox_to_anchor = (0.5, -0.5), fontsize=  plt.rcParams['font.size']-1)
# %%
plt.rcParams['font.family'] = 'Sans-serif'
plt.rcParams['font.size'] = 8
mpl.rcParams["axes.axisbelow"] = True
fontsize_label = 10


xspace = 70
yspace = 120

lineplot_x = 180
lineplot_y = 90
roc_x = 160
roc_y = 160

row = lineplot_y+yspace+roc_y
col = lineplot_x*2+xspace

fig = plt.figure(figsize=(col/100, row/100))
gsfig = gridspec.GridSpec(
    row, col, 
    left=0, right=1, bottom=0,
    top=1, wspace=1, hspace=1)

gs1 = gsfig[0:lineplot_y+yspace*0, 0:lineplot_x]
ax1 = fig.add_subplot(gs1)

gs2 = gsfig[0:lineplot_y+yspace*0, lineplot_x+xspace:lineplot_x*2+xspace]
ax2 = fig.add_subplot(gs2)

gs3 = gsfig[lineplot_y+yspace:lineplot_y+yspace+roc_y, 0:roc_x]
ax3 = fig.add_subplot(gs3)

draw_cumulative_auroc_lineplot(dict_train_ranking_to_auroc, n_tottop, ax1, mark_level = None, mark_name = None, ytick_scale = 0.02)
ax1.text(-0.3, 0.97, "A", transform = ax1.transAxes,
          size = fontsize_label, weight = "bold")

draw_cumulative_auroc_lineplot(dict_test_ranking_to_auroc, n_tottop, ax2, mark_level = auroc_fullmodel, mark_name = f"Full CpGs AUROC (365 CpGs)\n(AUROC: {auroc_fullmodel:.3f})", ytick_scale = 0.05)
ax2.text(-0.3, 0.97, "B", transform = ax2.transAxes,
          size = fontsize_label, weight = "bold")

draw_roc_prc(pd.read_csv(path_test_roc_format.format(ntop = 1), sep = '\t'), "Answer", "Proba_Case", "Prediction", name_case = "Case", draw_fig = True, draw_roc_curve = True, draw_prc_curve = False, overlap_figures = True, draw_baseline = True, mark_best_threshold = False, choose_balanced = False, choose_best_acc = False, label_name = "Top1 CpG site\n", name_control = "Control", ax = ax3, kwargs_line = {"linewidth" : 1.7, "color" : "r"}, kwargs_marker = {"s" : 35, "marker" : 'X'})
ax3.set_xticks(np.linspace(0, 1, 6))
ax3.set_yticks(np.linspace(0, 1, 6))
ax3.text(-0.3*lineplot_x/roc_x, 0.97, "C", transform=ax3.transAxes, 
            size=fontsize_label, weight='bold')

plt.savefig(f"{path_save_prefix}.png", dpi = 650, bbox_inches = "tight")
plt.savefig(f"{path_save_prefix}.pdf", dpi = 650, bbox_inches = "tight")
plt.show()
plt.close()
# %%
