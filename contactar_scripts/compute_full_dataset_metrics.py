# This script computes the metrics configured in SCORING list by training general models with different combination of features
# and evaluating them in indoor, outdoor and the complete dataset.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import mlflow.sklearn
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import sem
from numpy import mean
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

KFOLD=5

N_REPEATS = 5

MODEL_PARAMS = {
    "SVC" : [{'kernel': ['rbf'], 'C': [1, 10, 100]}, {'kernel': ['poly'], 'degree': [2], 'C': [1, 10, 100]}],
    "LogisticRegression" : [{'penalty': ['l2'], 'C': [1,5,10,100]}],
    "RandomForestClassifier": [{'n_estimators': [100], 'max_depth': [4, 6, 10, 14], 'criterion': ['entropy']}],
}

POSITION_FEATURES = ['mean', 'trim_mean', 'median', 'q1', 'q3', 'max', 'min']
DISPERTION_FEATURES = ['std', 'dis1', 'dis2', 'range', 'q3q1_ran']
OTHER_FEATURES = ['conteo', 'skew', 'kur']
combinations = []
# combinaciones de 1 feature
for feature in POSITION_FEATURES:
    combinations.append([feature])
# combinaciones de 2 features
for feature_1 in POSITION_FEATURES:
    for feature_2 in DISPERTION_FEATURES:
        combinations.append([feature_1, feature_2])
# combinaciones de 2 features
for feature_1 in POSITION_FEATURES:
    for feature_2 in OTHER_FEATURES:
        combinations.append([feature_1, feature_2])
# combinaciones de 3 features
for feature_1 in POSITION_FEATURES:
    for feature_2 in DISPERTION_FEATURES:
        for feature_3 in OTHER_FEATURES:
            combinations.append([feature_1, feature_2, feature_3])

SCORING = ['accuracy', 'precision', 'recall', 'f1']

########################################################################################################################

def get_model(model_str, random_state=42):
    if model_str == "SVC":
        return SVC(random_state=random_state)
    elif model_str == "RandomForestClassifier":
        return RandomForestClassifier(random_state=random_state)
    elif model_str == "LogisticRegression":
        return linear_model.LogisticRegression(random_state=random_state)
    elif model_str == "GaussianNB":
        return GaussianNB()
    elif model_str == "KMeans":
        return KMeans(random_state=random_state)

df = pd.read_csv("../contactar_datasets/inout-resumen-v3.csv")
df = df.dropna()
df = df[df['std'] != 0]
df = df[df['range'] > 1]
df['distance'] = df['distance'].apply(lambda x: 1 if x < 2 else 0)
df['environment'] = df['environment'].apply(lambda x: 1 if x == "indoor" else 0)
df['distance-environment'] = df['distance'].astype(str) + df['environment'].astype(str)

for model_str, parameters_to_explore in MODEL_PARAMS.items():

    for scoring in SCORING:

        experiment = mlflow.create_experiment("full_dataset" + "_" + model_str + "_" + scoring)

        model = get_model(model_str)

        for combination in combinations:

            print(model_str + "_" + scoring + "_" + "-".join(combination))

            mlflow.start_run(experiment_id=experiment)

            for i, feature in enumerate(combination):
                mlflow.log_param("feature_" + str(i+1), feature)

            # agrego esto solo para poder separar despues indoor de outdoor, pero remuevo esta columna antes de pasarsela al fit
            comb = combination + ['environment']

            # 1) Obtengo el mejor modelo para esta combinacion en mixed-indoor-outdoor
            labels = np.array(df['distance'])

            features = np.array(df[comb])

            X_train_with_env, X_test_with_env, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=df[['distance-environment']], random_state=42)
            # elimino la columna de environment
            X_train = X_train_with_env[:, :-1]
            X_test = X_test_with_env[:, :-1]

            scaling = MinMaxScaler().fit(X_train)
            X_train = scaling.transform(X_train)
            X_test = scaling.transform(X_test)

            print("getting best model")
            clf = GridSearchCV(model, parameters_to_explore, scoring=scoring, cv=KFOLD, n_jobs=-1)
            clf.fit(X_train, y_train)

            # 2) Separo el dataset de Train en indoor y outdoor
            X_train_indoor = []
            X_train_outdoor = []
            y_train_indoor = []
            y_train_outdoor = []
            for i, x in enumerate(X_train_with_env):
                if x[-1] == 1:
                    X_train_indoor.append(X_train[i])
                    y_train_indoor.append(y_train[i])
                elif x[-1] == 0:
                    X_train_outdoor.append(X_train[i])
                    y_train_outdoor.append(y_train[i])
                else:
                    raise Exception("error")

            print("computing metrics for best model")
            # 3) Utilizo el mejor modelo para hacer predicciones en indoor y predicciones en outdoor por separado
            # obtengo métricas utilizando RepeatedKFold
            cv = RepeatedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=1)
            scores = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
            scores_indoor = cross_val_score(clf, X_train_indoor, y_train_indoor, scoring=scoring, cv=cv, n_jobs=-1)
            scores_outdoor = cross_val_score(clf, X_train_outdoor, y_train_outdoor, scoring=scoring, cv=cv, n_jobs=-1)

            mlflow.log_param('scores', scores)
            mlflow.log_param('scores_mean', mean(scores))
            mlflow.log_param('scores_se', sem(scores))

            mlflow.log_param('scores_indoor', scores_indoor)
            mlflow.log_param('scores_indoor_mean', mean(scores_indoor))
            mlflow.log_param('scores_indoor_se', sem(scores_indoor))

            mlflow.log_param('scores_outdoor', scores_outdoor)
            mlflow.log_param('scores_outdoor_mean', mean(scores_outdoor))
            mlflow.log_param('scores_outdoor_se', sem(scores_outdoor))

            y_true, y_pred = y_test, clf.predict(X_test)
            if scoring == "accuracy":
                mlflow.log_metric('score_in_test', metrics.accuracy_score(y_true, y_pred))
            elif scoring == 'precision':
                mlflow.log_metric('score_in_test', metrics.precision_score(y_true, y_pred))
            elif scoring == 'recall':
                mlflow.log_metric('score_in_test', metrics.recall_score(y_true, y_pred))
            elif scoring == 'f1':
                mlflow.log_metric('score_in_test', metrics.f1_score(y_true, y_pred))

            mlflow.end_run()
