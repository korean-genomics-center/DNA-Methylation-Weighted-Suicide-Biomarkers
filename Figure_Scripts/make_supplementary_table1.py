#%%
import os
from pathlib import Path
os.chdir(str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from joblib import Parallel, delayed


#%%
path_significant = "./Data/significant_marker_list.Weighting_Count_Suicidal_Event_FillNA1_Squared.fdr_05.methdiff_5.count_90perc.txt"
path_bootstrap_format = "**Not provided due to large file size**"
path_annot = "./Data/significant_marker_list.weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.region.annotation.Annotatr.organized.enst_converted.tsv"

path_save = "./Table/Supplementary_Table1.xlsx"

n_seed = 1000
#%%
table_sig = pd.read_csv(path_significant, sep = '\t')
table_annot = pd.read_csv(path_annot, sep = '\t')
#%%
table_sig["PosName"] = table_sig["chr"] + '_' + table_sig["start"].astype(str) + '_' + table_sig["end"].astype(str)
# %%
def get_stats_from_dmp_result(path_dmp, list_posnames):
    table_dmp = pd.read_csv(path_dmp, sep = '\t')
    
    table_dmp["posname"] = table_dmp[["chr", "start"]].apply(lambda row: f"{row['chr']}_{row['start']}_{row['start']}", axis = 1)
    
    table_dmp_interest = table_dmp[table_dmp["posname"].isin(list_posnames)]
    
    list_dict_infos = table_dmp_interest.apply(lambda row: {"methdiff":row["meth.diff"], "fdr":row["qvalue"]}, axis = 1).to_list()
    
    dict_posname_to_info = dict(zip(table_dmp_interest["posname"], list_dict_infos))
    return dict_posname_to_info

with Parallel(n_jobs = 50) as parallel:
    list_dict_posname_to_info = parallel(
        delayed(get_stats_from_dmp_result)(
            path_bootstrap_format.format(seed = seed),
            set(table_sig["PosName"].to_list())
        ) for seed in range(n_seed)
    )
# %%
table_sig["Mean_methdiff"] = table_sig["PosName"].apply(lambda name:
    np.mean(list(map(lambda dict_info: dict_info[name]["methdiff"], list_dict_posname_to_info)))    
)
table_sig["Min_methdiff"] = table_sig["PosName"].apply(lambda name:
    np.min(list(map(lambda dict_info: dict_info[name]["methdiff"], list_dict_posname_to_info)))    
)
table_sig["Max_methdiff"] = table_sig["PosName"].apply(lambda name:
    np.max(list(map(lambda dict_info: dict_info[name]["methdiff"], list_dict_posname_to_info)))    
)
table_sig["Median_fdr"] = table_sig["PosName"].apply(lambda name:
    np.median(list(map(lambda dict_info: dict_info[name]["fdr"], list_dict_posname_to_info)))    
)
table_sig["Min_fdr"] = table_sig["PosName"].apply(lambda name:
    np.min(list(map(lambda dict_info: dict_info[name]["fdr"], list_dict_posname_to_info)))    
)
table_sig["Max_fdr"] = table_sig["PosName"].apply(lambda name:
    np.max(list(map(lambda dict_info: dict_info[name]["fdr"], list_dict_posname_to_info)))    
)
#%%
dict_name_to_n_sig = dict()
for name in table_sig["PosName"]:
    n_cnt = 0
    for dict_posname_to_info in list_dict_posname_to_info:
        methdiff = dict_posname_to_info[name]["methdiff"]
        fdr = dict_posname_to_info[name]["fdr"]
        
        if abs(methdiff) > 5 and fdr < 0.05:
            n_cnt += 1
    dict_name_to_n_sig[name] = n_cnt
# %%
table_sig["n_sig"] = table_sig["PosName"].apply(dict_name_to_n_sig.__getitem__)
# %%
table_sig["perc_sig"] = table_sig["n_sig"] / n_seed * 100
# %%
dict_posname_to_annot = dict()

table_annot_dropdup = table_annot.drop_duplicates(subset = ["PosName", "Gene_Symbol_Converted", "Gene_Attrib"])
for posname in table_annot_dropdup["PosName"].unique():
    table_annot_dropdup_pos = table_annot_dropdup[table_annot_dropdup["PosName"] == posname]
    
    list_gene_to_attrib_pair = list()
    for genename in table_annot_dropdup_pos["Gene_Symbol_Converted"].unique():
        table_annot_dropdup_pos_gene = table_annot_dropdup_pos[table_annot_dropdup_pos["Gene_Symbol_Converted"] == genename]
        list_attrib = sorted(table_annot_dropdup_pos_gene["Gene_Attrib"].to_list())
        attrib_show = '/'.join(list_attrib)
        
        genename_show = f"{genename} ({attrib_show})"
        list_gene_to_attrib_pair.append(genename_show)
    dict_posname_to_annot[posname] = ", ".join(list_gene_to_attrib_pair)
#%%
table_sig["gene"] = table_sig["PosName"].apply(dict_posname_to_annot.__getitem__)
#%%
table_sig_save = table_sig[["chr", "start", "direction", "perc_sig", "gene", 'Mean_methdiff', 'Min_methdiff', 'Max_methdiff', 'Median_fdr', 'Min_fdr', 'Max_fdr']]

table_sig_save_rename = table_sig_save.rename(
    columns = {
        "chr" : "Chromosome",
        "start" : "Position",
        "direction" : "Direction",
        "Mean_methdiff" : "Mean Weighted Methylation Difference (%)",
        "Min_methdiff" : "Minimum Weighted Methylation Difference (%)",
        "Max_methdiff" : "Maximum Weighted Methylation Difference (%)",
        "Median_fdr" : "Median FDR",
        "Min_fdr":  "Minimum FDR",
        "Max_fdr" : "Maximum FDR",
        "perc_sig" : "Percentage Significance (%)",
        "gene" : "Gene Annotation"
    }
)
# %%
table_sig_save_rename.to_excel(path_save, engine = "openpyxl", index = False)
# %%
