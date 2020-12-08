import torch
class Arguments():
    def __init__(self):
        self.batch_size = 2
        self.test_batch_size = 1000
        self.epochs = 4
        self.agg_epochs = 5
        self.lr = 0.01
        self.momentum = 0.5
        self.device = "cuda:0"
        self.seed = 1
        self.log_interval = 50

        # data distribution ratios and node details
        self.number_of_nodes = 2
        self.data_distribution = [0.5, 0.5]

        # data location details
        self.csv_location = '../csv/'
        self.data_location = '../x-ray/'
        self.agg_model_path = '../aggregated_model/'
