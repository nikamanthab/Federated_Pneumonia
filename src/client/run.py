import parser
import json
import os
from dataloader import getNumSamples, getTrainLoader
from httpcalls import getConnection, getModel, sendModel
from modelloader import loadModel
import torch
from torch import optim
import train

args = parser.getArguments()
NumOfSamples = getNumSamples(args)
args['no_of_samples'] = NumOfSamples

# Connecting to the aggregator server
agg_url = args['agg_ip']+':'+args['agg_port']
agg_url = 'http://' + agg_url

serverargs = getConnection(agg_url, args).json()
args['image_dim'] = (serverargs['image_height'], serverargs['image_width'])
model_path = os.path.join(args['model_location'], args['node_name']+'.pt')

while(True):# change to epoch iteration max
    # Gets model from the aggregator and stores in local system
    getModel(agg_url, model_path)
    
    # Creating train loader
    train_loader = getTrainLoader(args)
    
    # Train loop
    local_model = loadModel(model_path).to(args['device'])
    optimizer = optim.Adam(local_model.parameters(), lr=args['lr'])
    # for epoch in range(1, args['epochs'] + 1):
    #         train.train(args=args, model=local_model, \
    #             train_loader=train_loader, \
    #             optimizer=optimizer, epoch=epoch)
    
    torch.save(local_model, model_path)
    # Send model to the aggregator
    sendModel(agg_url, model_path, args)

    break