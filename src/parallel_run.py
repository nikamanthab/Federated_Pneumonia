
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
import log

loop = asyncio.get_event_loop()

async def runOnNode(node, args, model, train_loader):
    x_model = copy.deepcopy(model)
    x_model.send(node)
    optimizer = optim.Adam(x_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train.train(args=args, model=x_model, 
                    train_loader=train_loader, 
                    optimizer=optimizer, epoch=epoch)
    x_model.get()
    return x_model

async def collectModels(nodelist, args, model, FLdataloaders):
    task_list = []
    for idx, dataloader in enumerate(FLdataloaders):
        task_list.append(loop.create_task(runOnNode(nodelist[idx], args, model, dataloader)))
    await asyncio.wait(task_list)
    return [m.result() for m in task_list]

  
def runTrainParallel(nodelist, datasample_count, args, FLdataloaders, test_loader):   
    agg_model = None
    # if os.path.exists(os.path.join(args.agg_model_path, 'agg_model.pt')) == True:
    #     agg_model = torch.load(os.path.join(args.agg_model_path, 'agg_model.pt')).to(args.device)
    # else:
    agg_model = args.model().to(args.device)
    #wandb
    if args.wandb == True:
        logger = log.initialize_wandb(agg_model)
    else:
        logger = None
    
    for agg_epoch in range(1,args.agg_epochs+1):

        # Disttributed training
        print("Aggregation Epoch Number:", agg_epoch)
        
        node_model_list = loop.run_until_complete(collectModels(nodelist, args, agg_model, FLdataloaders))
        # print(node_model_list)
        # print(datasample_count)
        
        # Aggregation
        model_tuple_array = []
        for idx in range(len(FLdataloaders)):
            # node_model = torch.load(os.path.join(args.agg_model_path, nodelist[idx].id + '.pt'))
            node_model = node_model_list[idx]
            model_tuple_array.append((node_model, datasample_count[idx]))
        # print(model_tuple_array)
        
        agg_model = args.aggregator(model_data=model_tuple_array, args=args)

        torch.save(agg_model, os.path.join(args.agg_model_path, 'agg_model.pt'))
        print("Saving to: "+ os.path.join(args.agg_model_path, 'agg_model.pt'))

        # Testing
        test.test(args, agg_model, args.device, test_loader, logger)
