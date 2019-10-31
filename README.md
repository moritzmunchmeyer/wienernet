# wienernet
Wiener filtering flat CMB maps with a Neural Network 

This is the code for the paper "Fast Wiener filtering of CMB maps with Neural Networks" (https://arxiv.org/abs/1905.05846), accepted to the NeurIPS 2019 workshop "Machine Learning and the Physical Sciences". 

Requirements:
- tensorflow
- quicklens

Environment variables:
- Make sure PYTHONPATH includes the mlcmb folder.

To run an example:

1. Edit the config mlcmb/config/config_128_t_35muK.ini. There set the datapath to where you want to store the datasets and networks. Create the two subfolders mentioned in the config.

2. Create a data set for this config by running python trainingdata.py ~/mlcmb/mlcmb/config_128_t_35muK.ini. This takes a while as it will Winer filter the test data also.

3. Train on the data by running python train.py ~/mlcmb/mlcmb/config_128_t_35muK.ini

4. After training generate evaluation metrics python eval.py ~/mlcmb/mlcmb/config_128_t_35muK.ini

5. View quality control plots with the notebook notebooks/wiener_results_t.ipynb