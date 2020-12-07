
import torch.optim as optim
import time
import train
import test
import models
import os
import torch
import aggregator
import syft as sy
import copy
import asyncio
loop = asyncio.get_event_loop()

async def runOnNode(node, args, model, train_loader): 
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    x_model = copy.deepcopy(model)
    x_model.send(node)
    for epoch in range(1, args.epochs + 1):
        train.train(args = args, model = model, train_loader = train_loader, optimizer = optimizer, epoch = epoch)
    x_model.get()
    return x_model

async def collectModels(nodelist, args, model, FLdataloaders):
    task_list = []
    for idx, dataloader in enumerate(FLdataloaders):
        task_list.append(loop.create_task(runOnNode(nodelist[idx], args, model, dataloader)))
    await asyncio.wait(task_list)
    return task_list

  
def runTrainParallel(nodelist, datasample_count, args, FLdataloaders, test_loader):
    for agg_epoch in range(1,args.agg_epochs+1):
        model = None
        if os.path.exists(os.path.join(args.agg_model_path, 'agg_model.pt')) == True:
            model = torch.load(os.path.join(args.agg_model_path, 'agg_model.pt')).to(args.device)
        else:
            model = models.CNN_basic.TwoLayerNet().to(args.device)

        # Disttributed training
        print("Aggregation Epoch Number:", agg_epoch)
        
        node_model_list = loop.run_until_complete(collectModels(nodelist, args, model, FLdataloaders))
                
        # Aggregation
        model_tuple_array = []
        for idx in range(len(FLdataloaders)):
            # node_model = torch.load(os.path.join(args.agg_model_path, nodelist[idx].id + '.pt'))
            node_model = node_model_list[idx]
            model_tuple_array.append((node_model, datasample_count[idx]))
        agg_model = aggregator.fed_avg_aggregator(model_tuple_array)
        torch.save(agg_model, os.path.join(args.agg_model_path, 'agg_model.pt'))

        # Testing
        test.test(args, model, args.device, test_loader)

