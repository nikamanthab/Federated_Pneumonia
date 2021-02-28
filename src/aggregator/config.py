# import torch
# import models
# import aggregator
import argparse

def Arguments():
    parser = argparse.ArgumentParser(description='Server module.')
    parser.add_argument('--architecture', type=str, default='TwoLayerNet', \
        help='TwoLayerNet, ResNeXt50, ResNet18, VGGNet, AlexNet')
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--agg_epochs', type=float, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--image_width', type=int, default=28)

    parser.add_argument('--aggregator', type=str, default='fedavg', \
        help="fedavg, comed, geomed")
    
    parser.add_argument('--agg_iterations', type=int, default=50)
    parser.add_argument('--agg_optimizer', type=str, default='Adam',\
        help="Adam, SGD supported")
    parser.add_argument('--agg_optim_lr', type=float, default=0.01)
    parser.add_argument('--agg_optim_momentum', type=float, default=0.1)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--num_of_nodes', type=int, default=2)
    parser.add_argument('--test_csv', type=str, default='../../csv/test_folder.csv')
    parser.add_argument('--data_location', type=str, default='../../x-ray/')
    parser.add_argument('--aggregated_model_location', type=str, default='../../aggregated_model/')
    parser.add_argument('--labels', type=str, default='NORMAL, PNEUMONIA')
    parser.add_argument('--smpc', type=bool, default=False)
    args = parser.parse_args()

    labelstr = args.labels.split(',')
    labels = [s.strip() for s in labelstr]

    default_vals = {"architecture" :  ["TwoLayerNet", "ResNeXt50", "ResNet18", "VGGNet", "AlexNet"],
                    "aggregator"   :  ["fedavg", "comed", "geomed"],
                    "agg_optimizer":  ["Adam", "SGD supported"]
                    }

    while(args.architecture not in default_vals["architecture"]):
        print("Wrong entry for architecture : ", args.architecture)
        print("Valid Entries:", default_vals["architecture"])
        args.architecture = input("Enter Again :")
    
    while(args.aggregator not in default_vals["aggregator"]):
        print("Wrong entry for aggregator : ", args.aggregator)
        print("Valid Entries:", default_vals["aggregator"])
        args.aggregator = input("Enter Again :")

    while(args.agg_optimizer not in default_vals["agg_optimizer"]):
        print("Wrong entry for agg_optimizer : ", args.agg_optimizer)
        print("Valid Entries:", default_vals["agg_optimizer"])
        args.agg_optimizer = input("Enter Again :")

    



    cmdargs = {
        "architecture": args.architecture,
        "test_batch_size": args.test_batch_size,
        "agg_epochs": args.agg_epochs,
        "device": args.device,
        "momentum": args.momentum,
        "log_interval": args.log_interval,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "aggregator": args.aggregator,
        "agg_iterations": args.agg_iterations,
        "agg_optim_lr": args.agg_optim_lr,
        "agg_optim_momentum": args.agg_optim_momentum,
        "wandb": args.wandb,
        "num_of_nodes": args.num_of_nodes,
        "test_csv": args.test_csv,
        "data_location": args.data_location,
        "aggregated_model_location": args.aggregated_model_location,
        "labels": labels,
        "image_dim": (args.image_height, args.image_width),
        "smpc": args.smpc
    }

    return cmdargs

# class Arguments():
#     def __init__(self):
#         self.batch_size = 8
#         self.test_batch_size = 1000
#         self.epochs = 4
#         self.agg_epochs = 10
#         self.lr = 0.01
#         self.momentum = 0.5
#         self.device = "cuda:0"
#         self.seed = 1
#         self.image_dim = (28, 28)
#         self.log_interval = 50
#         self.model = models.TwoLayerNet

#         # Aggregator details
#         self.aggregator = aggregator.Geometric_Median
#         # Aggregator parameters
#         self.agg_params = {
#             "iterations": 10,
#             "optimizer": "Adam", # Adam, SGD
#             "optim_lr": 0.01,
#             "optim_momentum": 0.1
#         }

#         #wandb on/off
#         self.wandb = False

#         # Data distribution ratios and node details
#         self.number_of_nodes = 2
#         self.data_distribution = [0.5, 0.5]

#         # Data location details
#         self.csv_location = '../csv/'
#         self.data_location = '../x-ray/'
#         self.agg_model_path = '../aggregated_model/'
