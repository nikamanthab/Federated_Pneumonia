# Federated Learning Strategies for Image Classification
Exploring FL aggregation and robustness on distributed model training.

Data-set: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Instructions to run:
- Install the requirements - torch, pandas, wandb, flask
- Clone the repo in multiple systems/ run multiple instances with changed input parameters
- Download the dataset and form the following folder structure: x-ray/(train or test)/(NORMAL or PNEUMONIA)
- Start the server and start the client instances with appropriate parameters like node_name and train_csv
- CNN Architecture can be altered in the models.py and modelloader.py
- Execute the run.py file in the program folder

### Support And Future Development:
- Supports GPU in main branch
- For simulation with PySyft refer syft branch

### Command to run:
    SERVER:
        cd src/aggregator
        export FLASK_APP=server.py
        python server.py
        
        usage: server.py [-h] [--architecture ARCHITECTURE]
                 [--test_batch_size TEST_BATCH_SIZE] [--agg_epochs AGG_EPOCHS]
                 [--device DEVICE] [--momentum MOMENTUM]
                 [--log_interval LOG_INTERVAL] [--image_height IMAGE_HEIGHT]
                 [--image_width IMAGE_WIDTH] [--aggregator AGGREGATOR]
                 [--agg_iterations AGG_ITERATIONS]
                 [--agg_optimizer AGG_OPTIMIZER] [--agg_optim_lr AGG_OPTIM_LR]
                 [--agg_optim_momentum AGG_OPTIM_MOMENTUM] [--wandb WANDB]
                 [--num_of_nodes NUM_OF_NODES] [--test_csv TEST_CSV]
                 [--data_location DATA_LOCATION]
                 [--aggregated_model_location AGGREGATED_MODEL_LOCATION]
                 [--labels LABELS]

        Server module.

        optional arguments:
          -h, --help            show this help message and exit
          --architecture ARCHITECTURE
                                TwoLayerNet, ResNeXt50, ResNet18
          --test_batch_size TEST_BATCH_SIZE
          --agg_epochs AGG_EPOCHS
          --device DEVICE
          --momentum MOMENTUM
          --log_interval LOG_INTERVAL
          --image_height IMAGE_HEIGHT
          --image_width IMAGE_WIDTH
          --aggregator AGGREGATOR
                                fedavg, codem, geomed
          --agg_iterations AGG_ITERATIONS
          --agg_optimizer AGG_OPTIMIZER
                                Adam, SGD supported
          --agg_optim_lr AGG_OPTIM_LR
          --agg_optim_momentum AGG_OPTIM_MOMENTUM
          --wandb WANDB
          --num_of_nodes NUM_OF_NODES
          --test_csv TEST_CSV
          --data_location DATA_LOCATION
          --aggregated_model_location AGGREGATED_MODEL_LOCATION
          --labels LABELS
    CLIENT:
        cd src/client
        python run.py

        usage: run.py [-h] [--node_name NODE_NAME] [--agg_ip AGG_IP]
                  [--agg_port AGG_PORT] [--train_batch_size TRAIN_BATCH_SIZE]
                  [--epochs EPOCHS] [--lr LR] [--device DEVICE]
                  [--momentum MOMENTUM] [--log_interval LOG_INTERVAL]
                  [--train_csv TRAIN_CSV] [--data_location DATA_LOCATION]
                  [--wandb WANDB] [--model_location MODEL_LOCATION]
                  [--labels LABELS]

        Client module.

        optional arguments:
          -h, --help            show this help message and exit
          --node_name NODE_NAME
          --agg_ip AGG_IP
          --agg_port AGG_PORT
          --train_batch_size TRAIN_BATCH_SIZE
          --epochs EPOCHS
          --lr LR
          --device DEVICE
          --momentum MOMENTUM
          --log_interval LOG_INTERVAL
          --train_csv TRAIN_CSV
          --data_location DATA_LOCATION
          --wandb WANDB
          --model_location MODEL_LOCATION
          --labels LABELS
      Example:
        System 1 / Terminal 1:
        python run.py --node_name node_0 --train_csv ../../csv/train_0.csv
        
        System 2 / Terminal 2:
        python run.py --node_name node_1 --train_sv ../../csv/train_1.csv

## Scores
| Model  | Number of nodes | Distribution | Epochs before aggregation | Aggregation_iter | Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Two layer CNN  | 2 nodes | 0.5, 0.5 | 4 epochs | 5 iters | 0.8477 |
