import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

loss_fn = CrossEntropyLoss()
def test(args, model, test_loader, logger=None):
    model.eval()
    loss = 0
    total=0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(args['device']), target.to(args['device'])
        output = model(data)
#             import pdb; pdb.set_trace()
        test_loss = loss_fn(output, target)
        total += test_loss.item()
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
        correct += pred.eq(target.view_as(pred)).sum().item()

    total /= len(test_loader.dataset)
    if(logger!=None):
        logger({"loss": total, "accuracy": 100. * correct / len(test_loader.dataset)})
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))