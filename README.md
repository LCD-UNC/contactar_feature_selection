# Contactar Feature Selection

This repository contains the datasets and scripts needed to obtain the results provided on the paper "Feature Selection for Proximity Estimation in COVID-19 Contact Tracing Appsbased on Bluetooth Low Energy (BLE)" presented in "Pervasive and Mobile Computing" journal in the Special Issue on IoT for Fighting COVID-19.

## Execution

1. Create python environment: virtualenv --python=python3.8 venv
2. Activate environment: source venv/bin/activate
3. Install requirements: pip install -r requirements.txt 
4. Execute the scripts contained in contactar_scripts folder:

-compute_features.py: computes the features by means of rssi aggregations over each run of each experiment.

-compute_full_dataset_metrics.py: computes the metrics configured in SCORING list by training general models with different combination 	of features and evaluating them in indoor, outdoor and the complete dataset.

-plot_metrics.py: plots the metrics of the N_best models in different environments: indoor/outdoor.

-plot_feature_variability.py: plots the variability of the features as the number of samples is increased.

-plot_crossmodels.py: plots the potencial gain in score when using specific (environment aware) models versus when using general (environment unaware) models.

5. Results are saved in contactar_results folder and plots are saved in contactar_plots folder.






