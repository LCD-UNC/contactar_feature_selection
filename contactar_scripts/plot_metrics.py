# This script plots the metrics of the N_best models in different environments: indoor/outdoor

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N_best = 5
metric = "accuracy"
input_folder = "../contactar_results"
output_folder = "../contactar_plots"
environment = "indoor"

def get_flatten_df(file, model_type, metric, attribute_sort, attribute_flatten, model_env, N_best):
    features = ["feature_1", "feature_2", "feature_3"]
    df = pd.read_csv(file)
    df.feature_1.fillna("", inplace=True)
    df.feature_2.fillna("", inplace=True)
    df.feature_3.fillna("", inplace=True)

    ddict = {'q3q1_ran': 'iqr', 'conteo': 'cnt', 'trim_mean': 'tmean'}

    # add features and N_features columns
    def add_features_cols(row):
        features = []
        ft1 = row["feature_1"]
        ft2 = row["feature_2"]
        ft3 = row["feature_3"]

        if ft1:
            if ft1 in ddict:
                features.append(ddict[ft1])
            else:
                features.append(ft1)
        if ft2:
            if ft2 in ddict:
                features.append(ddict[ft2])
            else:
                features.append(ft2)
        if ft3:
            if ft3 in ddict:
                features.append(ddict[ft3])
            else:
                features.append(ft3)

        row["features"] = ("-").join(features)
        row["N_features"] = len(features)
        features.clear()
        return row

    df = df.apply(add_features_cols, axis=1)

    # keep N_best models for each N_features group
    g = df.groupby(["N_features"]).apply(lambda x: x.sort_values([attribute_sort], ascending=False)).reset_index(
        drop=True)
    g = g.groupby('N_features').head(N_best)

    # flatten scores to get 1 score in each row
    df2 = pd.DataFrame(columns=['features', 'N_features', metric, 'model'])
    for k, v in g.iterrows():
        scores = [float(i) for i in v[attribute_flatten][1:-1].replace(",", "").split()]
        for score in scores:
            df2 = df2.append(
                {'features': v['features'], 'N_features': v['N_features'], metric: score, 'model': model_env,
                 'model_type': model_type}, ignore_index=True)

    return df2


full_SVC = input_folder + "/full_dataset_SVC_" + metric + ".csv"
full_LR = input_folder + "/full_dataset_LogisticRegression_" + metric + ".csv"
full_RF = input_folder + "/full_dataset_RandomForestClassifier_" + metric + ".csv"

general_indoor_svc = get_flatten_df(full_SVC, 'SVC', metric, 'scores_indoor_mean', 'scores_indoor', 'general_indoor',
                                    N_best)
general_outdoor_svc = get_flatten_df(full_SVC, 'SVC', metric, 'scores_outdoor_mean', 'scores_outdoor',
                                     'general_outdoor', N_best)
df_SVC = pd.concat([general_indoor_svc, general_outdoor_svc])

general_indoor_lr = get_flatten_df(full_LR, 'LR', metric, 'scores_indoor_mean', 'scores_indoor', 'general_indoor',
                                   N_best)
general_outdoor_lr = get_flatten_df(full_LR, 'LR', metric, 'scores_outdoor_mean', 'scores_outdoor', 'general_outdoor',
                                    N_best)
df_LR = pd.concat([general_indoor_lr, general_outdoor_lr])

general_indoor_rf = get_flatten_df(full_RF, 'RF', metric, 'scores_indoor_mean', 'scores_indoor', 'general_indoor',
                                   N_best)
general_outdoor_rf = get_flatten_df(full_RF, 'RF', metric, 'scores_outdoor_mean', 'scores_outdoor', 'general_outdoor',
                                    N_best)
df_RF = pd.concat([general_indoor_rf, general_outdoor_rf])

sns.set_theme()
# sns.set_style("whitegrid")

dfs = [df_LR, df_SVC, df_RF]
MODEL_TYPES = ['LR', 'SVC', 'RF']
dict_model_types = {'LR': 'Logistic Regression', 'SVC': 'Support Vector Machine', 'RF': 'Random Forest'}
colors = ['b', 'g', 'r']

text_height = 0.13
x_shift = -0.15
xlabelsize = 10
means_text_size = 8
vertical_offset = 0.2  # acc-in=0.2 , acc-out=0.13

fig, ax = plt.subplots(3, 3, figsize=(10, 12))
rotation = 90

for i, model_type in enumerate(MODEL_TYPES):

    df = dfs[i]
    d = df[(df.model_type == model_type) & (df.model == 'general_' + environment)]

    d2 = d.copy(deep=True)
    d2 = d2[d2.N_features == 1]
    means = d2.groupby(["features"])[metric].mean()
    order_1 = means.sort_values(ascending=True).index
    smeans = [round(m, 3) for m in means.sort_values(ascending=True).values]
    g = sns.boxplot(data=d[d.N_features == 1], x="features", y=metric, ax=ax[i, 0], color=colors[i], order=order_1)
    g.set(xlabel=None)
    g.set(ylabel=metric + "\n" + dict_model_types[model_type])
    if i == 0:
        g.set(title="Single feature")
    # set x ticks texts in other position
    xticklabels = [t.get_text() for t in g.get_xticklabels()]
    for x in range(len(xticklabels)):
        g.text(x + x_shift, text_height, xticklabels[x], fontsize=xlabelsize, rotation=rotation)
    # set means as text
    for xtick in g.get_xticks():
        g.text(xtick, smeans[xtick] + vertical_offset, smeans[xtick], horizontalalignment='center',
               size=means_text_size, color='dimgrey', weight='semibold')
    g.set(xticklabels=[])

    d2 = d.copy(deep=True)
    d2 = d2[d2.N_features == 2]
    means = d2.groupby(["features"])[metric].mean()
    order_2 = means.sort_values(ascending=True).index
    smeans = [round(m, 3) for m in means.sort_values(ascending=True).values]
    g = sns.boxplot(data=d[d.N_features == 2], x="features", y=metric, ax=ax[i, 1], color=colors[i], order=order_2)
    g.set(xlabel=None)
    g.set(ylabel=None)
    g.set(yticklabels=[])
    if i == 0:
        g.set(title="Double feature")
    # set x ticks texts in other position
    xticklabels = [t.get_text() for t in g.get_xticklabels()]
    for x in range(len(xticklabels)):
        g.text(x + x_shift, text_height, xticklabels[x], fontsize=xlabelsize, rotation=rotation)
    # set means as text
    for xtick in g.get_xticks():
        g.text(xtick, smeans[xtick] + vertical_offset, smeans[xtick], horizontalalignment='center',
               size=means_text_size, color='dimgrey', weight='semibold')
    g.set(xticklabels=[])

    d2 = d.copy(deep=True)
    d2 = d2[d2.N_features == 3]
    means = d2.groupby(["features"])[metric].mean()
    order_3 = means.sort_values(ascending=True).index
    smeans = [round(m, 3) for m in means.sort_values(ascending=True).values]
    g = sns.boxplot(data=d[d.N_features == 3], x="features", y=metric, ax=ax[i, 2], color=colors[i], order=order_3)
    g.set(xlabel=None)
    g.set(ylabel=None)
    g.set(yticklabels=[])
    if i == 0:
        g.set(title="Triple feature")
    # set x ticks texts in other position
    xticklabels = [t.get_text() for t in g.get_xticklabels()]
    for x in range(len(xticklabels)):
        g.text(x + x_shift, text_height, xticklabels[x], fontsize=xlabelsize, rotation=rotation)
    # set means as text
    for xtick in g.get_xticks():
        g.text(xtick, smeans[xtick] + vertical_offset, smeans[xtick], horizontalalignment='center',
               size=means_text_size, color='dimgrey', weight='semibold')
    g.set(xticklabels=[])

for m, subplot in np.ndenumerate(ax):
    subplot.set_ylim(0.11, 1.19)

fig.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(output_folder + "/" + environment + ".eps", format='eps', dpi=900)


