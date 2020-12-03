# This script plots the potencial gain in score when using specific (environment aware) models versus when using general (environment unaware) models.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

input_folder = "../contactar_results"
output_folder = "../contactar_plots"

def get_flatten_df(file, model_type, metric, attribute_sort, attribute_flatten, model_env, N_best):
    features = ["feature_1", "feature_2", "feature_3"]
    df = pd.read_csv(file)
    df.feature_1.fillna("", inplace=True)
    df.feature_2.fillna("", inplace=True)
    df.feature_3.fillna("", inplace=True)

    ddict = {'q3q1_ran': 'iqr', 'conteo': 'cnt', 'trim_mean': 'trmean'}

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

N_best = 5
metric = "accuracy"

full_SVC = input_folder + "/full_dataset_SVC_" + metric + ".csv"
full_LR = input_folder + "/full_dataset_LogisticRegression_" + metric + ".csv"
full_RF = input_folder + "/full_dataset_RandomForestClassifier_" + metric + ".csv"

# Obtengo lo mejores N_best modelos para cada environment

nbest_general_indoor_svc = get_flatten_df(full_SVC, 'SVC', metric, 'scores_indoor_mean', 'scores_indoor',
                                          'general_indoor', N_best)
nbest_general_outdoor_svc = get_flatten_df(full_SVC, 'SVC', metric, 'scores_outdoor_mean', 'scores_outdoor',
                                           'general_outdoor', N_best)
nbest_df_SVC = pd.concat([nbest_general_indoor_svc, nbest_general_outdoor_svc])

nbest_general_indoor_lr = get_flatten_df(full_LR, 'LR', metric, 'scores_indoor_mean', 'scores_indoor', 'general_indoor',
                                         N_best)
nbest_general_outdoor_lr = get_flatten_df(full_LR, 'LR', metric, 'scores_outdoor_mean', 'scores_outdoor',
                                          'general_outdoor', N_best)
nbest_df_LR = pd.concat([nbest_general_indoor_lr, nbest_general_outdoor_lr])

nbest_general_indoor_rf = get_flatten_df(full_RF, 'RF', metric, 'scores_indoor_mean', 'scores_indoor', 'general_indoor',
                                         N_best)
nbest_general_outdoor_rf = get_flatten_df(full_RF, 'RF', metric, 'scores_outdoor_mean', 'scores_outdoor',
                                          'general_outdoor', N_best)
nbest_df_RF = pd.concat([nbest_general_indoor_rf, nbest_general_outdoor_rf])

# Obtengo todos los modelos para cada environment

general_indoor_svc = get_flatten_df(full_SVC, 'SVC', metric, 'scores_indoor_mean', 'scores_indoor', 'general_indoor',
                                    10000)
general_outdoor_svc = get_flatten_df(full_SVC, 'SVC', metric, 'scores_outdoor_mean', 'scores_outdoor',
                                     'general_outdoor', 10000)
df_SVC = pd.concat([general_indoor_svc, general_outdoor_svc])

general_indoor_lr = get_flatten_df(full_LR, 'LR', metric, 'scores_indoor_mean', 'scores_indoor', 'general_indoor',
                                   10000)
general_outdoor_lr = get_flatten_df(full_LR, 'LR', metric, 'scores_outdoor_mean', 'scores_outdoor', 'general_outdoor',
                                    10000)
df_LR = pd.concat([general_indoor_lr, general_outdoor_lr])

general_indoor_rf = get_flatten_df(full_RF, 'RF', metric, 'scores_indoor_mean', 'scores_indoor', 'general_indoor',
                                   10000)
general_outdoor_rf = get_flatten_df(full_RF, 'RF', metric, 'scores_outdoor_mean', 'scores_outdoor', 'general_outdoor',
                                    10000)
df_RF = pd.concat([general_indoor_rf, general_outdoor_rf])

indoor_lr = pd.concat([nbest_general_indoor_lr, general_outdoor_lr], ignore_index=True, sort=False)
nbest_features = nbest_general_indoor_lr.features.unique()
indoor_lr = indoor_lr[indoor_lr.features.isin(nbest_features)]

indoor_svc = pd.concat([nbest_general_indoor_svc, general_outdoor_svc], ignore_index=True, sort=False)
nbest_features = nbest_general_indoor_svc.features.unique()
indoor_svc = indoor_svc[indoor_svc.features.isin(nbest_features)]

indoor_rf = pd.concat([nbest_general_indoor_rf, general_outdoor_rf], ignore_index=True, sort=False)
nbest_features = nbest_general_indoor_rf.features.unique()
indoor_rf = indoor_rf[indoor_rf.features.isin(nbest_features)]

outdoor_lr = pd.concat([nbest_general_outdoor_lr, general_indoor_lr], ignore_index=True, sort=False)
nbest_features = nbest_general_outdoor_lr.features.unique()
outdoor_lr = outdoor_lr[outdoor_lr.features.isin(nbest_features)]

outdoor_svc = pd.concat([nbest_general_outdoor_svc, general_indoor_svc], ignore_index=True, sort=False)
nbest_features = nbest_general_outdoor_svc.features.unique()
outdoor_svc = outdoor_svc[outdoor_svc.features.isin(nbest_features)]

outdoor_rf = pd.concat([nbest_general_outdoor_rf, general_indoor_rf], ignore_index=True, sort=False)
nbest_features = nbest_general_outdoor_rf.features.unique()
outdoor_rf = outdoor_rf[outdoor_rf.features.isin(nbest_features)]

dfa = indoor_lr.groupby(['N_features', 'model'])['accuracy'].mean().reset_index()
dfa['model_type'] = "LR"
dfb = indoor_svc.groupby(['N_features', 'model'])['accuracy'].mean().reset_index()
dfb['model_type'] = "SVC"
dfc = indoor_rf.groupby(['N_features', 'model'])['accuracy'].mean().reset_index()
dfc['model_type'] = "RF"
df_indoor = pd.concat([dfa, dfb, dfc])

dfd = outdoor_lr.groupby(['N_features', 'model'])['accuracy'].mean().reset_index()
dfd['model_type'] = "LR"
dfe = outdoor_svc.groupby(['N_features', 'model'])['accuracy'].mean().reset_index()
dfe['model_type'] = "SVC"
dff = outdoor_rf.groupby(['N_features', 'model'])['accuracy'].mean().reset_index()
dff['model_type'] = "RF"
df_outdoor = pd.concat([dfd, dfe, dff])

df_indoor.head(10)

sns.set_theme()

palette2 = sns.color_palette("dark")
palette1 = sns.color_palette("tab10")

blue1 = palette1[0]
blue2 = palette2[0]
green1 = palette1[2]
green2 = palette2[2]
red1 = palette1[3]
red2 = palette2[3]

fig, axs = plt.subplots(2, 3, figsize=(12, 6))

lower_offset = 0.08
upper_offset = 0.03
x_shift = 0.16
decimals = 3

df1 = df_indoor[df_indoor.model_type == "LR"]
df2 = df_outdoor[df_outdoor.model_type == "LR"]
ax = axs[0, 0]
for n_features in range(1, 4):
    bar_height = df1[(df1.N_features == n_features) & (df1.model == "general_indoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=blue1)
    ax.text(n_features - x_shift, bar_height + upper_offset, round(bar_height, decimals), color='black')
for n_features in range(1, 4):
    bar_height = df2[(df2.N_features == n_features) & (df2.model == "general_indoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=blue2)
    ax.text(n_features - x_shift, bar_height - lower_offset, round(bar_height, decimals), color='white')
ax.set_title("Logistic Regression")
ax.set_ylabel("Accuracy\nindoor")
ax.get_xaxis().set_visible(False)

df1 = df_indoor[df_indoor.model_type == "SVC"]
df2 = df_outdoor[df_outdoor.model_type == "SVC"]
ax = axs[0, 1]
for n_features in range(1, 4):
    bar_height = df1[(df1.N_features == n_features) & (df1.model == "general_indoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=green1)
    ax.text(n_features - x_shift, bar_height + upper_offset, round(bar_height, decimals), color='black')
for n_features in range(1, 4):
    bar_height = df2[(df2.N_features == n_features) & (df2.model == "general_indoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=green2)
    ax.text(n_features - x_shift, bar_height - lower_offset, round(bar_height, decimals), color='white')
ax.set_title("Support Vector Machine")
ax.get_xaxis().set_visible(False)

df1 = df_indoor[df_indoor.model_type == "RF"]
df2 = df_outdoor[df_outdoor.model_type == "RF"]
ax = axs[0, 2]
for n_features in range(1, 4):
    bar_height = df1[(df1.N_features == n_features) & (df1.model == "general_indoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=red1)
    ax.text(n_features - x_shift, bar_height + upper_offset, round(bar_height,decimals), color='black')
for n_features in range(1, 4):
    bar_height = df2[(df2.N_features == n_features) & (df2.model == "general_indoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=red2)
    ax.text(n_features - x_shift, bar_height - lower_offset, round(bar_height, decimals), color='white')
ax.set_title("Random Forest")
ax.get_xaxis().set_visible(False)

####################################

df1 = df_indoor[df_indoor.model_type == "LR"]
df2 = df_outdoor[df_outdoor.model_type == "LR"]
ax = axs[1, 0]
for n_features in range(1, 4):
    bar_height = df2[(df2.N_features == n_features) & (df2.model == "general_outdoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=blue1)
    ax.text(n_features - x_shift, bar_height + upper_offset, round(bar_height, decimals), color='black')
for n_features in range(1, 4):
    bar_height = df1[(df1.N_features == n_features) & (df1.model == "general_outdoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=blue2)
    ax.text(n_features - x_shift, bar_height - lower_offset, round(bar_height, decimals), color='white')
ax.set_xlabel("Features count")
ax.set_ylabel("Accuracy\noutdoor")

df1 = df_indoor[df_indoor.model_type == "SVC"]
df2 = df_outdoor[df_outdoor.model_type == "SVC"]
ax = axs[1, 1]
for n_features in range(1, 4):
    bar_height = df2[(df2.N_features == n_features) & (df2.model == "general_outdoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=green1)
    ax.text(n_features - x_shift, bar_height + upper_offset, round(bar_height, decimals), color='black')
for n_features in range(1, 4):
    bar_height = df1[(df1.N_features == n_features) & (df1.model == "general_outdoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=green2)
    ax.text(n_features - x_shift, bar_height - lower_offset, round(bar_height, decimals), color='white')
ax.set_xlabel("Features count")

df1 = df_indoor[df_indoor.model_type == "RF"]
df2 = df_outdoor[df_outdoor.model_type == "RF"]
ax = axs[1, 2]
for n_features in range(1, 4):
    bar_height = df2[(df2.N_features == n_features) & (df2.model == "general_outdoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=red1)
    ax.text(n_features - x_shift, bar_height + upper_offset, round(bar_height, decimals), color='black')
for n_features in range(1, 4):
    bar_height = df1[(df1.N_features == n_features) & (df1.model == "general_outdoor")]['accuracy'].values[0]
    ax.bar(n_features, bar_height, color=red2)
    ax.text(n_features - x_shift, bar_height - lower_offset, round(bar_height, decimals), color='white')
ax.set_xlabel("Features count")

for m, subplot in np.ndenumerate(axs):
    subplot.set_ylim(0.2, 1.0)

plt.savefig(output_folder + "/" + "crossmodels" + ".eps", format='eps', dpi=900)



