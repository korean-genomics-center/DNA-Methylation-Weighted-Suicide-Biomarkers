#%%
import os
from pathlib import Path
os.chdir(str(Path(__file__).parents[1]))

import pandas as pd

#%%
path_kegg = "./Data/Gene_Enrichment/gene_enrich_weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.region.annotation.Annotatr.KEGG.csv"
path_gobp = "./Data/Gene_Enrichment/gene_enrich_weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.region.annotation.Annotatr.GOBP.csv"
path_gocc = "./Data/Gene_Enrichment/gene_enrich_weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.region.annotation.Annotatr.GOCC.csv"

path_save = "./Table/Supplementary_Table3.xlsx"

#%%
table_kegg = pd.read_csv(path_kegg)
table_gobp = pd.read_csv(path_gobp)
table_gocc = pd.read_csv(path_gocc)

#%%
def modify_gene_enrichment_result(table_enrich):
    table_enrich["Pathway_id"] = table_enrich["Pathway"].apply(lambda val: val.split()[0].replace("Path", "KEGG"))
    table_enrich["Pathway_name"] = table_enrich["Pathway"].apply(lambda val: ' '.join(val.split()[1:]).lower())
    table_enrich["Pathway_name"] = table_enrich["Pathway_name"].apply(lambda val: val[0].upper() + val[1:])
    table_enrich["Genes"] = table_enrich["Genes"].apply(lambda val: ", ".join(val.split()))
    return table_enrich

table_kegg["Pathway_db"] = "KEGG"
table_gobp["Pathway_db"] = "GO Biological Process"
table_gocc["Pathway_db"] = "GO Cellular Component"

table_kegg_mod = modify_gene_enrichment_result(table_kegg)
table_gobp_mod = modify_gene_enrichment_result(table_gobp)
table_gocc_mod = modify_gene_enrichment_result(table_gocc)

table_pathway = pd.concat([table_kegg_mod, table_gobp_mod, table_gocc_mod]) 
#%%
table_pathway_save = table_pathway[[
    "Pathway_db",
    "Pathway_id",
    "Pathway_name",
    "nGenes",
    "Pathway Genes",
    "Fold Enrichment",
    "Enrichment FDR",
    "Genes"
]]
# %%
table_pathway_save_rename = table_pathway_save.rename(
    columns = {
        "Pathway_db" : "Pathway Database",
        "Pathway_id" : "Pathway ID",
        "Pathway_name" : "Pathway Name",
        "nGenes" : "Number of Related Genes",
        "Pathway Genes" : "Number of Pathway Genes",
        "Genes" : "Gene List"
    }
)
# %%
table_pathway_save_rename.to_excel(path_save, engine = "openpyxl", index = False)
# %%
