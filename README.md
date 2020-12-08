# Federated Learning Strategies for Image Classification
Exploring FL aggregation and robustness on distributed model training.
Data-set: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Instructions to run:
- Install the requirements - torch==1.4, syft==0.2.9, pandas
- Download the dataset and form the following folder structure: x-ray/(train or test)/(NORMAL or PNEUMONIA)
- Change the necessary parameters in program/config.py file
- CNN Architecture can be altered in the models/CNN_basic.py
- Execute the run.py file in the program folder


    python program/run.py

## Scores
| Model  | Number of nodes | Distribution | Epochs before aggregation | Aggregation_iter | Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Two layer CNN  | 2 nodes | 0.5, 0.5 | 4 epochs | 5 iters | 0.80 |
