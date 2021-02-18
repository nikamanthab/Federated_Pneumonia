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


parallel_run.runTrainParallel(nodelist, datasample_count, args, FLdataloaders, testloader)