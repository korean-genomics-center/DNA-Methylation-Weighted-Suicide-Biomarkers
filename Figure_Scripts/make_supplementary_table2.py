#%%
import os
from pathlib import Path
os.chdir(str(Path(__file__).parents[1]))

import pandas as pd
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import fdrcorrection

#%%
path_cpg = "./Data/Raw_CpG_Values.suicidal_biomarker.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.tsv"
path_meta = "**Not provided due to sensitive information**"

path_dmp = "./Data/significant_marker_list.Weighting_Count_Suicidal_Event_FillNA1_Squared.fdr_05.methdiff_5.count_90perc.1000_bootstrapping_summary.txt"

path_save = "./Table/Supplementary_Table2.xlsx"

#%%
table_cpg = pd.read_csv(path_cpg, sep = '\t')
table_meta = pd.read_csv(path_meta, sep = '\t')
table_dmp = pd.read_csv(path_dmp, sep = '\t')

#%%
table_meta_train = table_meta[table_meta["Sample_Split"] == "Train"]
table_meta_train_con = table_meta_train[table_meta_train["Sample_Group"] == "Normal"]
table_meta_train_case = table_meta_train[table_meta_train["Sample_Group"] == "Suicide_attempt"]

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
# %%
table_pos_relation["posname"] = table_pos_relation["chr"] + ':' + table_pos_relation["start"].astype(str)
table_dmp["posname"] = table_dmp["chr"] + ':' + table_dmp["start"].astype(str)
#%%
dict_posname_to_dir = dict(zip(table_dmp["posname"], table_dmp["Direction"]))
dict_posname_to_methdiff = dict(zip(table_dmp["posname"], table_dmp["Mean_WMD"]))
#%%
table_pos_relation["DMS_direction"] = table_pos_relation["posname"].apply(dict_posname_to_dir.__getitem__)
table_pos_relation["DMS_methdiff"] = table_pos_relation["posname"].apply(dict_posname_to_methdiff.__getitem__)
#%%
table_pos_relation_save = table_pos_relation[["chr", "start", "DMS_direction", "DMS_methdiff", "rho", "pval", "fdr"]]

# %%
table_pos_relation_save_rename = table_pos_relation_save.rename(
    columns = {
        "chr" : "Chromosome",
        "start" : "Position",
        "DMS_direction" : "Differential Methylation Analysis Direction",
        "DMS_methdiff" : "Differential Methylation Analysis Mean Weighted Difference (%)",
        "rho" : "Spearman Correlation Coefficient",
        "pval" : "Spearman Correlation P-value",
        "fdr" : "Spearman Correlation FDR"
    }
)
# %%
table_pos_relation_save_rename.to_excel(path_save, engine = "openpyxl", index = False)
# %%
