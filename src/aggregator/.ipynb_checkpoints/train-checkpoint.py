import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from tqdm import tqdm

loss_fn = CrossEntropyLoss()

def train(args, model, train_loader, optimizer, epoch, logger):
    batch_idx = 0
    total_loss = 0
    for (data, gt) in tqdm(train_loader):
        data, gt = data.to(args['device']), gt.to(args['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader.dataset)
    print("total_loss:", total_loss)

# def train(args, model, train_loader, optimizer, epoch, logger):
#     model.train()
#     total=0
#     length=0
#     for batch_idx, (data, target) in enumerate(train_loader): # <-- now it is a distributed dataset
#         data, target = data.to(args['device']), target.to(args['device'])
#         optimizer.zero_grad()
#         output = model(data)
# #         import pdb; pdb.set_trace()
#         loss = loss_fn(output,target)
# #         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()

#         if batch_idx % args['log_interval'] == 0:
#             # loss = loss.get() # <-- NEW: get the loss back
#             # print(loss, type(loss), loss.shape)
#             # if(logger!=None):
#             #     logger({"epoch": epoch , "loss": loss.item()})

#             total+=loss.item()
#             length+=1

#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * args['train_batch_size'], len(train_loader) * args['train_batch_size'], #batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

#     print("Average loss :",total/length)
#     if(logger!=None):
#         logger({"loss": total/length})