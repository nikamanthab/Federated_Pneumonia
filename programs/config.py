import torch
import models
import aggregator
class Arguments():
    def __init__(self):
        self.batch_size = 8
        self.test_batch_size = 1000
        self.epochs = 4
        self.agg_epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.device = "cuda:0"
        self.seed = 1
        self.image_dim = (28, 28)
        self.log_interval = 50
        self.model = models.TwoLayerNet

        # Aggregator details
        self.aggregator = aggregator.Geometric_Median
        # Aggregator parameters
        self.agg_params = {
            "iterations": 10,
            "optimizer": "Adam", # Adam, SGD
            "optim_lr": 0.01,
            "optim_momentum": 0.1
        }

        #wandb on/off
        self.wandb = False

        # Data distribution ratios and node details
        self.number_of_nodes = 2
        self.data_distribution = [0.5, 0.5]

        # Data location details
        self.csv_location = '../csv/'
        self.data_location = '../x-ray/'
        self.agg_model_path = '../aggregated_model/'
