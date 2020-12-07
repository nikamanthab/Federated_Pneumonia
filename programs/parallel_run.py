
import multiprocessing 
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
import torch.multiprocessing as mp
mp.set_start_method('spawn')
# class RunNode(Process): 
#     def __init__(self, node, args, model, train_loader): 
#         super(RunNode, self).__init__() 
#         self.node = node
#         self.args = args
#         self.model = model
#         print("sending...")
#         self.model.send(self.node)
#         print("sent...")
#         self.train_loader = train_loader
                 
#     def run(self): 
#         optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
#         print("hi2")
#         for epoch in range(1, self.args.epochs + 1):
#             print("h3")
#             train.train(args = self.args, model = self.model, device = self.args.device, 
#             train_loader = self.train_loader, optimizer = optimizer, epoch = epoch)
#         self.model.get()
#         torch.save(self.model, os.path.join(self.args.agg_model_path, self.node.id + '.pt'))

def runOnNode(node, args, model, train_loader): 
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    print("hi2")
    for epoch in range(1, args.epochs + 1):
        print("h3")
        train.train(args = args, model = model, device = args.device, 
        train_loader = train_loader, optimizer = optimizer, epoch = epoch)
    model.get()
    torch.save(model, os.path.join(args.agg_model_path, node.id + '.pt'))
  
def runTrainParallel(nodelist, datasample_count, args, FLdataloaders, test_loader):
    for agg_epoch in range(1,args.agg_epochs+1):
        model = None
        if os.path.exists(os.path.join(args.agg_model_path, 'agg_model.pt')) == True:
            model = torch.load(os.path.join(args.agg_model_path, 'agg_model.pt'))
        else:
            model = models.CNN_basic.TwoLayerNet()

        # Disttributed training
        print("Aggregation Epoch Number:", agg_epoch)
        node_processes = []
        print(nodelist)
        for idx, dataloader in enumerate(FLdataloaders):
            # process = RunNode(nodelist[idx], args, model, dataloader)
            x_model = copy.deepcopy(model)
            print("sending...")
            x_model.send(nodelist[idx])
            print("sent...")
            proc = mp.Process(target=runOnNode, args=(nodelist[idx], args, x_model, dataloader))
            proc.start()
            node_processes.append(proc)
            print(node_processes)
        print(node_processes)
        for idx in range(len(FLdataloaders)):
            print("h6...")
            print(node_processes)
            node_processes[idx].join()
        
        # Aggregation
        model_tuple_array = []
        for idx in range(len(FLdataloaders)):
            node_model = torch.load(os.path.join(args.agg_model_path, nodelist[idx].id + '.pt'))
            model_tuple_array.append((node_model, datasample_count[idx]))
        agg_model = aggregator.fed_avg_aggregator(model_tuple_array)
        torch.save(agg_model, os.path.join(args.agg_model_path, 'agg_model.pt'))

        # Testing
        test.test(args, model, args.device, test_loader)

