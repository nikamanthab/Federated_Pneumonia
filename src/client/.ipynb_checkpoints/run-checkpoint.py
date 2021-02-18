import config
import json
import os
from dataloader import getNumSamples, getTrainLoader
from httpcalls import getConnection, getModel, sendModel
from modelloader import loadModel
import torch
from torch import optim
import train
import log
from checks import checkModelEqual
from test import test
args = config.getArguments()
NumOfSamples = getNumSamples(args)
args['no_of_samples'] = NumOfSamples

# Connecting to the aggregator server
agg_url = args['agg_ip']+':'+args['agg_port']
agg_url = 'http://' + agg_url

serverargs = getConnection(agg_url, args)
local_agg_epoch = serverargs['current_agg_epoch']
args['image_dim'] = serverargs['image_dim']
model_path = os.path.join(args['model_location'], args['node_name']+'.pt')

# initialize wandb
if args['wandb'] == True:
    logger = log.initialize_wandb(args['node_name'])
else:
    logger = None


while(True):# change to aggregation epoch iteration max
    # Gets model from the aggregator and stores in local system
    getModel(agg_url, model_path, local_agg_epoch)
    
    # Creating train loader
    train_loader = getTrainLoader(args)
    # Train loop
    local_model = loadModel(model_path).to(args['device'])
    
    agg_model = torch.load('../../aggregated_model/agg_model.pt').to(args['device'])
    print("ModelCheck: ", checkModelEqual(local_model, agg_model))
    
    optimizer = optim.Adam(local_model.parameters(), lr=args['lr'])
    for epoch in range(1, args['epochs'] + 1):
            train.train(logger=logger ,
                args=args, model=local_model,      
                train_loader=train_loader, \
                optimizer=optimizer, epoch=epoch)                ###logger added
    
    test(args, local_model, train_loader, logger=None)
    torch.save(local_model, model_path)
    # Send model to the aggregator
    sendModel(agg_url, model_path, args)
    local_agg_epoch+=1
    # break
