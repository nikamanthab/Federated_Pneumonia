import torch
import torch.nn.functional as F


def train(args, model, device, train_loader, optimizer, epoch):
    print("hi4")
    model.train()
    print("hi5")
    for batch_idx, (data, target) in enumerate(train_loader): # <-- now it is a distributed dataset
        print(batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            loss = loss.get() # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size, #batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))