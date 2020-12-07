import torch
import os

import dataloader
import models
import parallel_run

#initialize syft
import syft as sy

# Get configuration parameters
import config
args = config.Arguments()

# Load and simulate distribution of data
FLdataloaders, datasample_count, nodelist = dataloader.getDataLoaders(args,sy)
testloader = dataloader.getTestLoader(args)

model = None
if os.path.exists(args.agg_model_path) == True:
    model = torch.load(args.agg_model_path)
else:
    model = models.CNN_basic.TwoLayerNet()

parallel_run.runTrainParallel(nodelist, model, datasample_count, args, FLdataloaders, testloader)









