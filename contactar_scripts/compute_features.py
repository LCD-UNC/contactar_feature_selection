# This script computes the features by means of rssi aggregations over each run of each experiment.

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# paths
path_in = '../contactar_datasets/indoorv4.csv'

path_out = '../contactar_datasets/outdoorv2.csv'

indoor = pd.read_csv(path_in)
indoor['environment'] = 'indoor'
outdoor = pd.read_csv(path_out)
outdoor['environment'] = 'outdoor'

print('indoor.tx_label.unique:', indoor.tx_label.unique(), 'len:', len(indoor))
print('outdoor.tx_label.unique:', outdoor.tx_label.unique(), 'len:', len(outdoor))

# drop tx_label == NN
indoor = indoor[indoor['tx_label'] != 'NN']
print('indoor.tx_label.unique:', indoor.tx_label.unique(), 'len:', len(indoor))

# muestras en indoor
print('NRO. DE MUESTRAS _INDOOR_ POR DISTANCIA')
mts_ = list(indoor.distance.unique())
tot = 0
for i in mts_:
  samples = 0
  mts = i
  exp_ = list(indoor.loc[indoor.distance==i].experiment.unique())
  for j in exp_:
    rx_ = indoor.loc[(indoor.distance==i)&(indoor.experiment==j)].rx_label.unique()
    for k in rx_:
      tx_ = indoor.loc[(indoor.distance==i)&(indoor.experiment==j)&(indoor.rx_label==k)].tx_label.unique()
      for kk in tx_:
        samples = samples + len(indoor.loc[(indoor.distance==i)&(indoor.experiment==j)&(indoor.rx_label==k)&(indoor.tx_label==kk)].run.unique())
        #for l in tx_:
         # if i == 1.0:
          #  print(j, i, k, l)
  print('mts:', i, 'muestras:', samples)
  tot = tot + samples
print('Total:', tot)

indoor.head(5)

# only for outdoor!!!
outdoor.loc[(outdoor['tx_label']=='A')&(outdoor['rx_label']=='C'), 'distance'] = 0
outdoor.loc[(outdoor['tx_label']=='C')&(outdoor['rx_label']=='A'), 'distance'] = 0
outdoor.loc[(outdoor['tx_label']=='B')&(outdoor['rx_label']=='D'), 'distance'] = 0
outdoor.loc[(outdoor['tx_label']=='D')&(outdoor['rx_label']=='B'), 'distance'] = 0

# muestras en indoor
print('NRO. DE MUESTRAS _INDOOR_ POR DISTANCIA')
mts_ = list(outdoor.distance.unique())
tot = 0
for i in mts_:
  samples = 0
  mts = i
  exp_ = list(outdoor.loc[outdoor.distance==i].experiment.unique())
  for j in exp_:
    rx_ = outdoor.loc[(outdoor.distance==i)&(outdoor.experiment==j)].rx_label.unique()
    for k in rx_:
      tx_ = outdoor.loc[(outdoor.distance==i)&(outdoor.experiment==j)&(outdoor.rx_label==k)].tx_label.unique()
      for kk in tx_:
        samples = samples + len(outdoor.loc[(outdoor.distance==i)&(outdoor.experiment==j)&(outdoor.rx_label==k)&(outdoor.tx_label==kk)].run.unique())
        #for l in tx_:
         # if i == 1.0:
          #  print(j, i, k, l)
  print('mts:', i, 'muestras:', samples)
  tot = tot + samples
print('Total:', tot)

outdoor.head(5)

inout_df = pd.concat([indoor, outdoor])
inout_df.tail(5)

"""# 3.MEDIDAS DE RESUMEN"""

# distancia p=21
def dis_p(x, p=21):
   d = sum(abs(x - np.median(x))**p/len(x))**(1/p)
   return d

# 
def dis1(x):
  d = sum(abs(x-np.mean(x)))/len(x)
  return d

#
def dis2(x):
  d = sum(abs(x-np.median(x)))/len(x)
  return d

# Summaty measures

df_stat = inout_df.groupby(['environment', 'distance', 'experiment', 'rx_label', 'tx_label', 'run']).agg(
    mean = pd.NamedAgg(column='rssi', aggfunc='mean'),
    trim_mean = pd.NamedAgg(column='rssi', aggfunc=lambda x: stats.trim_mean(x, 0.1)),
    median = pd.NamedAgg(column='rssi', aggfunc='median'),
    std = pd.NamedAgg(column='rssi', aggfunc='std'),
    #sum = pd.NamedAgg(column='rssi', aggfunc='sum'),
    dis1 = pd.NamedAgg(column='rssi', aggfunc=lambda x: dis1(x)),
    dis2 = pd.NamedAgg(column='rssi', aggfunc=lambda x: dis2(x)),    
    dis21 = pd.NamedAgg(column='rssi', aggfunc=lambda x: dis_p(x)),
    conteo = pd.NamedAgg(column='rssi', aggfunc=lambda x: len(x.unique())),
    cv = pd.NamedAgg(column='rssi', aggfunc=lambda x: np.std(x)/np.mean(x)), # coeficiente de variacion. 
    q1 = pd.NamedAgg(column='rssi', aggfunc=lambda x: np.quantile(x, .25)),
    q3 = pd.NamedAgg(column='rssi', aggfunc=lambda x: np.quantile(x, .75)),
    q3q1_ran = pd.NamedAgg(column='rssi', aggfunc=lambda x: np.quantile(x, .75)-np.quantile(x, .25)),
    max = pd.NamedAgg(column='rssi', aggfunc=max),
    min = pd.NamedAgg(column='rssi', aggfunc=min),
    #cvmax = pd.NamedAgg(column='rssi', aggfunc=lambda x: np.std(np.amax(x))/np.amax((x))), # como retorna 0, verificar.
    #stdmax = pd.NamedAgg(column='rssi', aggfunc=lambda x: np.std(np.amax(x))), # como retorna 0, verificar.
    range = pd.NamedAgg(column='rssi', aggfunc=lambda x: np.ptp(x)),
    skew = pd.NamedAgg(column='rssi', aggfunc=lambda x: stats.skew(x)), # (simetría)
    kur = pd.NamedAgg(column='rssi', aggfunc=lambda x: stats.kurtosis(x)) # (que tan puntiaguda)
    )

df_stat = df_stat.reset_index(level=[0,1,2,3,4])

df_stat.head(25)

len(df_stat)

distances = list(df_stat.distance.unique())
print('CANTIDAD DE ENTRADAS POR DISTANCIA')
for i in distances: print('distance:',i, 'total entradas:', df_stat[df_stat.distance == i].shape[0])

distances = list(df_stat.distance.unique())
print('CANTIDAD DE ENTRADAS _INTDOOR_ POR DISTANCIA')
for i in distances: print('distance:',i, 'total entradas:', df_stat.loc[(df_stat.environment=='indoor') & (df_stat.distance == i)].shape[0])

distances = list(df_stat.distance.unique())
print('CANTIDAD DE ENTRADAS _OUTDOOR_ POR DISTANCIA')
for i in distances: print('distance:',i, 'total entradas:', df_stat.loc[(df_stat.environment=='outdoor') & (df_stat.distance == i)].shape[0])

# save
df_stat.to_csv('../contactar_datasets/inout-resumen-v3.csv')

#fig, ax = plt.subplots(figsize = (15, 12))
#
# plt.figure(1)
#
# plt.subplot(221)
# sns.scatterplot(data=df_stat.loc[df_stat.environment=='indoor'], x='mean', y='dis1',
#                 hue='distance', palette='deep')
# #plt.title('Puntos de muestra de A y B')
#
# plt.subplot(222)
# sns.scatterplot(data=df_stat.loc[df_stat.environment=='indoor'], x='q3', y='dis1',
#                 hue='distance', palette='deep')
# #plt.title('Puntos de muestra por lote')
#
# plt.subplot(223)
# sns.scatterplot(data=df_stat.loc[df_stat.environment=='indoor'], x='median', y='dis2',
#                 hue='distance', palette='deep')
# #plt.title('Puntos de muestra de A y B')
#
# plt.subplot(224)
# sns.scatterplot(data=df_stat.loc[df_stat.environment=='indoor'], x='q3', y='dis2',
#                 hue='distance', palette='deep')
#
# plt.subplots_adjust(hspace=0.2)
# plt.show()