This repository contains the PyTorch implementation of the MedAI 2023 paper: XG-DTA: Drug-Target Affinity Prediction Based on Drug Molecular Graph and Protein Sequence combined with XLNet, available at: `https://ieeexplore.ieee.org/document/10403269`.

## Source code:
- create_data.py: creates data in pytorch format
- utils.py: contains the TestbedDataset and performance metrics for the data created by create_data.py.
- training.py: trains the XG-DTA model.
- models.py: contains model details.

## Step-by-step running:
### 0. Install Python libraries needed
The version of torch we use:
- torch 2.0.1+cu117
- torch-cluster 1.6.1+pt20cu117
- torch-geometric 2.1.0
- torch-scatter 2.1.1+pt20cu117
- torch-sparse 0.6.17+pt20cu117


### 1.Create data in pytorch format
Our data has been uploaded to `https://figshare.com/articles/dataset/data/25671039`. You need to download davis_test.csv, davis_train.csv, kiba_test.csv, and kiba_train.csv to the data folder.
```
python data_process.py
```
This will get the pre-training .pt file for the data

### 2.Train a prediction model with validation
```
python training.py
```
This returns the model achieving the best MSE for validation data throughout the training and performance results of the model on testing data. 


