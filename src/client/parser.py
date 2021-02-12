import argparse

# class ClientArgs:
#     def __init__(self):
#         self.args = getArguments()
#         self.node_name = self.args.node_name
#         self.agg_ip = self.args.agg_ip
#         self.agg_port = self.args.agg_port
#         self.batch_size = self.args.batch_size
#         self.epochs = self.args.epochs
#         self.lr = self.args.lr
#         self.device = self.args.device
#         self.momentum = self.args.momentum
#         self.log_interval = self.args.log_interval
#         self.csv_location = self.args.csv_location
#         self.data_location = self.args.data_location


def getArguments():
    parser = argparse.ArgumentParser(description='Client module.')
    parser.add_argument('--node_name', type=str, default='node_0')
    parser.add_argument('--agg_ip', type=str, default='localhost')
    parser.add_argument('--agg_port', type=str, default='5000')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=float, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--train_csv', type=str, default='../../csv/train.csv')
    parser.add_argument('--data_location', type=str, default='../../x-ray/')
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--model_location', type=str, default='../../node_model/')
    parser.add_argument('--labels', type=str, default='NORMAL, PNEUMONIA')
    args = parser.parse_args()
    # return args

    labelstr = args.labels.split(',')
    labels = [s.strip() for s in labelstr]

    cmdargs = {
        "node_name": args.node_name,
        "agg_ip": args.agg_ip,
        "agg_port": args.agg_port,
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": args.device,
        "momentum": args.momentum,
        "log_interval": args.log_interval,
        "train_csv": args.train_csv,
        "data_location": args.data_location,
        "model_location": args.model_location,
        "wandb": args.wandb,
        "labels": labels
    }
    return cmdargs