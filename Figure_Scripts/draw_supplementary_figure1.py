#%%
import os
from pathlib import Path
os.chdir(str(Path(__file__).parents[1]))

import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import Counter

#%%
path_meta = "**Not provided due to sensitive information**"

path_save_prefix = "./Figures/Supplementary_Figure1"

#%%
table_meta = pd.read_csv(path_meta, sep = '\t')
table_meta_train = table_meta[table_meta["Sample_Split"] == "Train"]
table_meta_train_case = table_meta_train[table_meta_train["Sample_Group"] == "Suicide_attempt"]

#%%
count_suicide_attempt = Counter(table_meta_train_case["Count_Suicidal_Event"])

#%%
plt.rcParams['font.family'] = 'Sans-serif'
plt.rcParams['font.size'] = 9
mpl.rcParams["axes.axisbelow"] = True

fig = plt.figure(figsize=(3, 2))

list_xticks = list(range(1, int(max(count_suicide_attempt.keys())+1)))
list_yval = list(map(lambda val: count_suicide_attempt[val], list_xticks))

plt.bar(list_xticks, list_yval, color = "gray")

for xtick, yval in zip(list_xticks, list_yval):
    plt.annotate(yval, (xtick, yval), ha = "center", va = "bottom")

plt.xticks(list_xticks, list_xticks)
plt.ylim(top = max(count_suicide_attempt.values()) * 1.15)
plt.xlabel("# of Suicide Attempts")
plt.ylabel("# of Individuals")
plt.savefig(f"{path_save_prefix}.png", dpi = 650, bbox_inches = "tight")
plt.savefig(f"{path_save_prefix}.pdf", dpi = 650, bbox_inches = "tight")
plt.show()
plt.close()
# %%
