# This script plots the variability of the features as the number of samples is increased

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_folder = "../contactar_plots"

POSITION_FEATURES = ['mean', 'trim_mean', 'median', 'q1', 'q3', 'max', 'min']
DISPERTION_FEATURES = ['std', 'dis1', 'dis2', 'range', 'q3q1_ran']
OTHER_FEATURES = ['conteo', 'skew', 'kur']

df = pd.read_csv("../contactar_datasets/indoor_varios_len_resumen-v1.csv")
df = df.dropna()
df = df[df['std'] != 0]
df = df[df['range'] > 1]
df['distance'] = df['distance'].apply(lambda x: 1 if x < 2 else 0)

df.columns = ['Unamed','distance', 'experiment', 'rx_label', 'tx_label', 'run',
       'mean', 'tmean', 'median', 'std', 'dis1', 'dis2', 'cnt', 'cv',
       'q1', 'q3', 'iqr', 'max', 'min', 'range', 'skew', 'kur', 'len']

df.len.unique()

lens = list(range(20,320,20))

df = df[df.len.isin(lens)]

sns.set_theme()

fig, axs = plt.subplots(3,3, figsize=(14,14))
ax=sns.boxplot(data=df, x="len", y="mean", ax=axs[0,0], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.get_xaxis().set_visible(False)
ax.set_title("mean")
ax.set_ylabel(None)

ax=sns.boxplot(data=df, x="len", y="min", ax=axs[0,1], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.get_xaxis().set_visible(False)
ax.set_title("min")
ax.set_ylabel(None)

ax=sns.boxplot(data=df, x="len", y="max", ax=axs[0,2], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.get_xaxis().set_visible(False)
ax.set_title("max")
ax.set_ylabel(None)

ax=sns.boxplot(data=df, x="len", y="std", ax=axs[1,0], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.get_xaxis().set_visible(False)
ax.set_title("std")
ax.set_ylabel(None)

ax=sns.boxplot(data=df, x="len", y="dis1", ax=axs[1,1], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.get_xaxis().set_visible(False)
ax.set_title("dis1")
ax.set_ylabel(None)

ax=sns.boxplot(data=df, x="len", y="dis2", ax=axs[1,2], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.get_xaxis().set_visible(False)
ax.set_title("dis2")
ax.set_ylabel(None)

ax=sns.boxplot(data=df, x="len", y="cnt", ax=axs[2,0], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title("cnt")
ax.set_ylabel(None)
ax.set_xlabel("Samples")

ax=sns.boxplot(data=df, x="len", y="skew", ax=axs[2,1], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title("skew")
ax.set_ylabel(None)
ax.set_xlabel("Samples")

ax=sns.boxplot(data=df, x="len", y="kur", ax=axs[2,2], color='b')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title("kur")
ax.set_ylabel(None)
ax.set_xlabel("Samples")

plt.savefig(output_folder + "/" + "n_samples" + ".eps", format='eps', dpi=900)



