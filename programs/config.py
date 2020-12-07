import torch
class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 20
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 1
        self.log_interval = 10
        self.save_model = False

        # data distribution ratios and node details
        self.number_of_nodes = 2
        self.data_distribution = [0.5, 0.5]

        # data location details
        self.csv_location = '../csv/'
        self.data_location = '../x-ray/'
        self.agg_model_path = '../aggregated_model/agg_model.pt'
