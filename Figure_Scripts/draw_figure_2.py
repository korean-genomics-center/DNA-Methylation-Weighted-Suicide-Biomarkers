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
from statsmodels.stats.multitest import fdrcorrection


#%%
##################################
# Panel A
path_cpg = "./Data/Raw_CpG_Values.suicidal_biomarker.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.tsv"
path_meta = "**Not provided due to sensitive information**"
path_dmp_bootstrap = "./Data/significant_marker_list.Weighting_Count_Suicidal_Event_FillNA1_Squared.fdr_05.methdiff_5.count_90perc.1000_bootstrapping_summary.txt"
path_annotation = "./Data/significant_marker_list.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.region.annotation.Annotatr.organized.enst_converted.tsv"

#%%
import pandas as pd

table_cpg = pd.read_csv(path_cpg, sep = '\t')
table_meta = pd.read_csv(path_meta, sep = '\t')

table_meta_train = table_meta[table_meta["Sample_Split"] == "Train"]
table_meta_train_con = table_meta_train[table_meta_train["Sample_Group"] == "Normal"]
table_meta_train_case = table_meta_train[table_meta_train["Sample_Group"] == "Suicide_attempt"]

from scipy.stats import spearmanr

dict_sa_sample_to_attempt = dict(zip(table_meta_train_case["Sample_ID"], table_meta_train_case["Count_Suicidal_Event"]))
table_pos_relation = pd.DataFrame(columns = ["chr", "start", "end", "rho", "pval"])

for ind, row in table_cpg.iterrows():
    dict_values = row.to_dict()
    list_samples = list(dict_sa_sample_to_attempt.keys())
    list_attempt = list(map(dict_sa_sample_to_attempt.__getitem__, list_samples))
    list_cpgs = list(map(dict_values.__getitem__, list_samples))
    spearman_result = spearmanr(list_attempt, list_cpgs)
    rho = spearman_result.correlation
    pval = spearman_result.pvalue
    table_pos_relation.loc[ind, :] = [dict_values["chr"], dict_values["start"], dict_values["end"], rho, pval]

_, fdr = fdrcorrection(table_pos_relation["pval"])
table_pos_relation["fdr"] = fdr
table_pos_relation_sig = table_pos_relation[table_pos_relation["fdr"] < 0.05]

table_cpg["posname"] = table_cpg.apply(lambda x : f"{x['chr']}:{x['start']}", axis = 1)

# %%
table_dmp = pd.read_csv(path_dmp_bootstrap, sep = '\t')
table_pos_relation_sig["posname"] = table_pos_relation_sig.apply(lambda x : f"{x['chr']}:{x['start']}", axis = 1)
table_dmp["posname"] = table_dmp.apply(lambda x : f"{x['chr']}:{x['start']}", axis = 1)

dict_posname_to_methdiff = dict(zip(table_dmp["posname"], table_dmp["Mean_WMD"]))
table_pos_relation_sig["mean_methdiff"] = table_pos_relation_sig["posname"].apply(dict_posname_to_methdiff.__getitem__)

table_pos_relation["posname"] = table_pos_relation.apply(lambda x : f"{x['chr']}:{x['start']}", axis = 1)
table_pos_relation["mean_methdiff"] = table_pos_relation["posname"].apply(dict_posname_to_methdiff.__getitem__)

table_annot = pd.read_csv(path_annotation, sep = '\t')
# %%
table_annot["posname"] = table_annot.apply(lambda x : f"{x['chr']}:{x['start']}", axis = 1)
dict_posname_to_annot = dict()
for _, row in table_annot.iterrows():
    posname = row["posname"]
    annot = row["Gene_Symbol_Converted"]
    if dict_posname_to_annot.get(posname) == None:
        dict_posname_to_annot[posname] = list()
    dict_posname_to_annot[posname].append(annot)
# %%
table_pos_relation_sig["Annotation"] = table_pos_relation_sig["posname"].apply(lambda posname : list(set(dict_posname_to_annot[posname])))
table_pos_relation["Annotation"] = table_pos_relation["posname"].apply(lambda posname : '/'.join(list(set(dict_posname_to_annot[posname]))))
# %%
import math
list_genes = list()
for genes in table_pos_relation_sig["Annotation"].to_list():
    list_genes.extend(genes)
    
list_genes = list(set(list(filter(lambda x : pd.notna(x), list_genes))))
# %%
from matplotlib import pyplot as plt
import seaborn as sns
from adjustText import adjust_text

plt.figure(figsize = (4.76, 2.8))
plt.rcParams['font.family'] = 'Sans-serif'
plt.rcParams['font.size'] = 8
fontsize_label = 10

table_pos_relation["fdr_sig"] = table_pos_relation["fdr"].apply(lambda x : x < 0.05)
table_pos_relation["is_hyper"] = table_pos_relation.apply(lambda row : int(row["mean_methdiff"] > 0), axis = 1)
table_pos_relation["fdr_sig_and_is_hyper"] = table_pos_relation.apply(lambda row : 0 if not row["fdr_sig"] else row["fdr_sig"] + row["is_hyper"], axis = 1)
sns.scatterplot(data = table_pos_relation, x = "mean_methdiff", y = "rho", hue = "fdr_sig_and_is_hyper",  size = "fdr_sig_and_is_hyper", size_order = [2, 1, 0], sizes = [6, 6, 3], hue_order = [2, 1, 0], palette = [(237/255,28/255,36/255), (0/255,102/255,160/255), (209/255,211/255,212/255)])
plt.axhline(0, linewidth = 1, linestyle = "--", color = "darkgray", zorder = -1)
plt.axvline(-5, linewidth = 1, linestyle = "--", color = "darkgray", zorder = -1)
plt.axvline(5, linewidth = 1, linestyle = "--", color = "darkgray", zorder = -1)
list_text = list()
for _, row in table_pos_relation_sig.iterrows():
    genename = '/'.join(sorted(row["Annotation"]))
    if genename in ["KCNC4/KCNC4-DT", "PHGDH", "ZNF767P"]:
        genename = {"KCNC4/KCNC4-DT" : "KCNC4"}.get(genename, genename)
        text_row = plt.text(row["mean_methdiff"], row["rho"], genename, weight = "bold", color = 'k', style = "italic", fontsize = plt.rcParams["font.size"]-2)
    else:
        # text_row = plt.text(row["mean_methdiff"], row["rho"], genename, fontsize = plt.rcParams["font.size"] - 8)
        continue
    list_text.append(text_row)
adjust_text(list_text, arrowprops = dict(arrowstyle = "-", color = 'k', lw = 0.5))
leg = plt.legend(title = "Significant\nCorrelation\n(FDR < 0.05)", loc = "center left", bbox_to_anchor = (1.01, 0.5), fontsize = plt.rcParams["font.size"]-1.5, title_fontsize = plt.rcParams["font.size"]-1.5)
leg.get_texts()[0].set_text("Yes (Hyper.)")
leg.get_texts()[1].set_text("Yes (Hypo.)")
leg.get_texts()[2].set_text("No")
leg.legendHandles[0]._sizes = [6]
leg.legendHandles[1]._sizes = [6]
leg.legendHandles[2]._sizes = [3]
# leg._legend_box.align = "left"
plt.ylim(-0.63, 0.63)
plt.xlabel("Mean(Weighted Mean Methylation Difference) (%)")
plt.ylabel("Correlation Coefficient")

plt.tight_layout()
plt.savefig("./Figures/Figure_2.pdf", dpi = 650, bbox_inches='tight')
plt.savefig("./Figures/Figure_2.png", dpi = 650, bbox_inches='tight')
plt.show()
plt.close()
# %%
