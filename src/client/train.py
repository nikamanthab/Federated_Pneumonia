import torch
import torch.nn.functional as F


def train(args, model, train_loader, optimizer, epoch, logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # <-- now it is a distributed dataset
        data, target = data.to(args['device']), target.to(args['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            # loss = loss.get() # <-- NEW: get the loss back
            # print(loss, type(loss), loss.shape)
            if(logger!=None):
                logger({"epoch": epoch , "loss": loss.item()})
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args['train_batch_size'], len(train_loader) * args['train_batch_size'], #batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))