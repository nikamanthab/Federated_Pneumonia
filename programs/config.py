import torch
import models
class Arguments():
    def __init__(self):
        self.batch_size = 8
        self.test_batch_size = 1000
        self.epochs = 4
        self.agg_epochs = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.device = "cuda:0"
        self.seed = 1
        self.image_dim = (28, 28)
        self.log_interval = 50
        self.model = models.TwoLayerNet

        # data distribution ratios and node details
        self.number_of_nodes = 1
        self.data_distribution = [1.0,]

        # data location details
        self.csv_location = '../csv/'
        self.data_location = '../x-ray/'
        self.agg_model_path = '../aggregated_model/'
