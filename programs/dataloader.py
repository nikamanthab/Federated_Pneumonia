import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import PIL

classes = {}
pt = 0
for i in pd.read_csv('../csv/train.csv')['label'].unique():
    classes[i] = pt
    pt+=1

def get_onehot(label):
    return classes[label]

class XDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.df.iloc[idx]['label']
        img_name = os.path.join(self.root_dir,label,str(self.df.iloc[idx]['image']))
        image = Image.open(img_name)
        image = PIL.ImageOps.grayscale(image)
        onehot = np.array(get_onehot(label))
        if self.transform:
            image = self.transform(image)
        return (image,onehot)

def getDataLoaders(args, sy):
    hook = sy.TorchHook(torch)
    number_of_nodes = args.number_of_nodes
    df = pd.read_csv(os.path.join(args.csv_location+'train.csv'))
    df = df.sample(frac=1)[:200]
    df_len = len(df)
    data_distribution = [int(df_len*ratio) for ratio in args.data_distribution]
    datasample_count = data_distribution
    for idx in range(1,len(data_distribution)):
        data_distribution[idx] = data_distribution[idx]+data_distribution[idx-1]
    data_distribution.insert(0, 0)
    df_list = []
    for cuts in range(1,len(data_distribution)):
        df_list.append(df[data_distribution[cuts-1]: data_distribution[cuts]])
    
    dataloader_list = []
    node_list = []
    node_counter = 0
    for frame in df_list:
        node = sy.VirtualWorker(hook, id="node"+str(node_counter))
        node_list.append(node)
        dataset = XDataset(frame, os.path.join(args.data_location, 'train'), transform=transforms.Compose(
            [transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]))
        data_loader = sy.FederatedDataLoader(
            dataset.federate((node,)),
            batch_size=args.batch_size, shuffle=True)
        dataloader_list.append(data_loader)
        node_counter+=1
    return dataloader_list, datasample_count, node_list

def getTestLoader(args):
    df_test = pd.read_csv(os.path.join(args.csv_location,'test.csv'))
    df_test = df_test.sample(frac=1)
    testdataset = XDataset(df_test,os.path.join(args.data_location,'test'), transform=transforms.Compose(
        [transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]))

    test_loader = torch.utils.data.DataLoader(
        testdataset,
        batch_size=args.test_batch_size, shuffle=True)
    return test_loader
