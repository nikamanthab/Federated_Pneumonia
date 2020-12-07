
import multiprocessing 
import torch.optim as optim
import time
import train
import test
import models
import os
import torch
import aggregator
  
  
class RunNode(multiprocessing.Process): 
    def __init__(self, node, args, model, train_loader): 
        super(RunNode, self).__init__() 
        self.node = node
        self.args = args
        self.model = model
        self.train_loader = train_loader
                 
    def run(self): 
        self.model.send(self.node)
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
        for epoch in range(1, self.args.epochs + 1):
            train.train(args = self.args, model = self.model, device = self.args.device, 
            train_loader = self.train_loader, optimizer = optimizer, epoch = epoch)
        self.model.get()
        torch.save(self.model, os.path.join(self.args.agg_model_path, self.node.id + '.pt'))
  
def runTrainParallel(nodelist, model, datasample_count, args, FLdataloaders, test_loader):
    for agg_epoch in range(1,args.agg_epochs+1):
        # Disttributed training
        print("Aggregation Epoch Number:", agg_epoch)
        node_processes = []
        for idx, dataloader in enumerate(FLdataloaders):
            process = RunNode(nodelist[idx], args, model, dataloader)
            node_processes.append(process)
            process.start()
        for idx in range(len(FLdataloaders)):
            node_processes[idx].join()
        
        # Aggregation
        model_tuple_array = []
        for idx in range(len(FLdataloaders)):
            node_model = torch.load(os.path.join(args.agg_model_path, nodelist[idx].id + 'pt'))
            model_tuple_array.append((node_model, datasample_count[idx]))
        aggregator.fed_avg_aggregator(model_tuple_array)

        # Testing
        test.test(args, model, args.device, test_loader)

