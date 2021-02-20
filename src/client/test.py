import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

loss_fn = CrossEntropyLoss()
def test(args, model, test_loader, logger=None):
    model.eval()
    test_loss = 0
    correct = 0
    total_pred = []
    total_target = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args['device']), target.to(args['device'])
            output = model(data)
            test_loss += loss_fn(output, target).item()
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_pred += list(pred.flatten().detach().cpu().numpy())
            total_target += list(target.detach().cpu().numpy())
            
    
    total_acc = accuracy_score(total_target, total_pred)
    precision = precision_score(total_target, total_pred)
    recall = recall_score(total_target, total_pred)
    f1 = f1_score(total_target, total_pred)
    
    
    test_loss /= len(test_loader.dataset)
    if(logger!=None):
        logger({"loss": test_loss, "accuracy": 100. * correct / len(test_loader.dataset)})
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Accuracy:'+str(total_acc)+'Precision:'+str(precision)+'\tRecall:'+str(recall)+'\tF1:'+str(f1))