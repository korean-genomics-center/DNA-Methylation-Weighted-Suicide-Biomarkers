#%%
import os
from pathlib import Path
os.chdir(str(Path(__file__).parents[1]))
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


#%%
###################################
path_1 = "./Data/Gene_Enrichment/gene_enrich_weight_count_suicidal_event_squared.fdr_05.methdiff_5.count_90perc.region.annotation.Annotatr.GOBP.csv"

name_1 = "GO (Biological Process)"

n_max_show = 15
Upper_first = True
no_pathway_id = True
adjust_showing_pathway_length = True
len_showing_adjust_kegg = 31
len_showing_adjust_go = 45

size_factor = "nGenes"
size_minmax = (20, 200)
size_show_level = 6

hue_factor = "Neg_Log10_FDR"
palette = "flare_r"

color_label_significant = "navy"
color_label_nonsignificant = "gray"

#%%
table_pathway1 = pd.read_csv(path_1)
table_pathway1["Pathway_id"] = table_pathway1["Pathway"].apply(lambda x : x.split()[0])
table_pathway1["Pathway_name"] = table_pathway1["Pathway"].apply(lambda x : ' '.join(x.split()[1:]))

if Upper_first:
    table_pathway1["Pathway_name"] = table_pathway1["Pathway_name"].apply(lambda x : x[0].upper() + x[1:])
    
if no_pathway_id:
    cols_show_pathway = ["Pathway_name"]
else:
    cols_show_pathway = ["Pathway_id", "Pathway_name"]        

table_pathway1["Pathway_show"] = table_pathway1[cols_show_pathway].apply(lambda row : ':'.join(row.to_list()), axis = 1)

def adjust_phrase_length(value, length_limit):
    if len(value) <= length_limit:
        return value
    import math
    ind_best = 0
    len_front = -math.inf
    list_values_split = value.split()
    for ind in range(len(list_values_split)+1):
        values_front = list_values_split[:ind]
        values_back = list_values_split[ind:]
        len_front_diff = len(' '.join(values_front)) - length_limit
        if len_front_diff > 0:
            break
        elif len_front_diff > len_front:
            ind_best = ind
            len_front = len_front_diff
    value_front = ' '.join(list_values_split[:ind_best])
    value_back = ' '.join(list_values_split[ind_best:])
    if len(value_back) > length_limit:
        value_back = adjust_phrase_length(value_back, length_limit)
    return value_front + '\n' + value_back

if adjust_showing_pathway_length:
    table_pathway1["Pathway_show"] = table_pathway1["Pathway_show"].apply(lambda val : adjust_phrase_length(val, len_showing_adjust_go))
    
def filter_table_with_fdr_fe(table, n_max_show, col_fdr = "Enrichment FDR", col_fe = "Fold Enrichment"):
    import math
    table["Neg_Log10_FDR"] = table[col_fdr].apply(lambda x : -math.log10(x))
    table_fdr_sig = table[table[col_fdr] < 0.05]
    table_fdr_sig["FDR_sig"] = 1
    table_fdr_nonsig = table[table[col_fdr] >= 0.05]
    table_fdr_nonsig["FDR_sig"] = 0
    
    list_tables_to_sort_fe = list()
    if table_fdr_sig.shape[0] > n_max_show:
        table_fdr_sig = table_fdr_sig.sort_values(by = col_fe, ascending = False)
        list_tables_to_sort_fe.append(table_fdr_sig.iloc[:n_max_show, :])    
    else:
        list_tables_to_sort_fe.append(table_fdr_sig)
        table_fdr_nonsig = table_fdr_nonsig.sort_values(by = col_fdr)
        list_tables_to_sort_fe.append(table_fdr_nonsig.iloc[:min(n_max_show-table_fdr_sig.shape[0], table_fdr_nonsig.shape[0]), :])    
    
    table_joined = pd.concat(list_tables_to_sort_fe)
    table_joined = table_joined.sort_values(by = ["FDR_sig", col_fe], ascending = False).reset_index(drop = True)
    return table_joined

table_pathway1_filtered = filter_table_with_fdr_fe(table_pathway1, n_max_show)

hue_min = min(table_pathway1_filtered[hue_factor].to_list())
hue_max = max(table_pathway1_filtered[hue_factor].to_list())
size_min = min(table_pathway1_filtered[size_factor].to_list())
size_max = max(table_pathway1_filtered[size_factor].to_list())


import math
hue_min_stepped = math.floor(hue_min / 0.5) * 0.5
hue_max_stepped = math.ceil(hue_max / 0.5) * 0.5
size_min_stepped = math.floor(size_min / 0.5) * 0.5
size_max_stepped = math.ceil(size_max / 0.5) * 0.5

# %%
#######################################
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib.cm import ScalarMappable

plt.rcParams['font.family'] = 'Sans-serif'
plt.rcParams['font.size'] = 8
fontsize_label = 10
ytick_extra_size = -1.1

fig = plt.figure(figsize = (4.76, 5))
row = 120
col = 100
gsfig = gridspec.GridSpec(
    row, col,
    left = 0, right = 1, bottom = 0,
    top = 1, wspace = 1, hspace = 1
)


gs1 = gsfig[0:92, 40:80]
ax1 = fig.add_subplot(gs1)

gs3 = gsfig[101:119, 7:44] 
ax3 = fig.add_subplot(gs3)

gs4 = gsfig[108:111, 50:85]
ax4 = fig.add_subplot(gs4)

gs5 = gsfig[100:120, 0:93]
ax5 = fig.add_subplot(gs5)

# Pathway1
sns.scatterplot(data = table_pathway1_filtered,
                x = "Fold Enrichment",
                y = "Pathway_show",
                hue = "Neg_Log10_FDR",
                hue_norm = (hue_min_stepped, hue_max_stepped),
                palette = palette,
                size = "nGenes",
                size_norm = (size_min_stepped, size_max_stepped),
                sizes = size_minmax,
                ax = ax1,
                legend = False,
                zorder = 3)
tick_colors = list(map({1:color_label_significant, 0:color_label_nonsignificant}.__getitem__, table_pathway1_filtered["FDR_sig"]))
for ticklabel, tickcolor in zip(ax1.get_yticklabels(), tick_colors):
    ticklabel.set_color(tickcolor)
    if tickcolor == color_label_significant:
        ticklabel.set_fontweight("bold")
    ticklabel.set_fontsize(plt.rcParams['font.size'] + ytick_extra_size)
ax1.grid(zorder = 1, linewidth = 0.5)
ax1.set_ylabel("")
ax1.set_title(name_1, fontsize = plt.rcParams["font.size"]+1)
ax1.set_xlim(-1, 34)
ax1.set_xticks(range(2, 33, 4))
# ax1.text(-0.9, 1.02, "A", transform = ax1.transAxes, size = fontsize_label, weight = "bold")

# Legend (Size)
list_handles = list()
list_labels = list()
size_step_show = math.ceil((size_max_stepped - size_min_stepped) / (size_show_level-1))
for step_multiplier in range(size_show_level):
    size_val = size_min_stepped + size_step_show * step_multiplier
    size_val_norm = size_val * ((size_minmax[1] - size_minmax[0]) / (size_max_stepped - size_min_stepped))
    list_handles.append(plt.scatter([],[], s = size_val_norm, c = "gray"))
    list_labels.append(str(int(size_val)))
ax3.legend(list_handles,
           list_labels,
           scatterpoints = 1,
           ncol = 3,
           title = "Number of Genes",
           labelspacing = 1.1,
           borderpad = 0,
           loc = "upper left",
           bbox_to_anchor = (0, 1),
           frameon = False)
ax3.set_axis_off()

# Legend (FDR)
import matplotlib as mpl

cmap = sns.color_palette(palette, as_cmap = True)
norm = mpl.colors.Normalize(vmin=hue_min_stepped, vmax=hue_max_stepped)
plt.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap),
             cax = ax4,
             orientation = "horizontal",
             label = "-Log10(FDR)",
             anchor = (0.5, 1),
             panchor = (1, 1))
ax4.xaxis.set_label_position("top")

# Draw Rectangle
rec = mpl.patches.Rectangle((0,0), 1, 1, linewidth = 1, edgecolor = "gray", facecolor = "none")
ax5.add_patch(rec)
ax5.set_axis_off()

gsfig.tight_layout(fig)
plt.savefig("./Figures/Supplementary_Figure4.pdf", dpi = 650, bbox_inches='tight')
plt.savefig("./Figures/Supplementary_Figure4.png", dpi = 650, bbox_inches='tight')
plt.show()
# %%
