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
import copy
import random

def get_onehot(labellist, label):
    return labellist.index(label)

class XDataset(Dataset):
    def __init__(self, df, labellist, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.labellist = labellist
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.df.iloc[idx]['label']
        folder = self.df.iloc[idx]['folder']
        img_name = os.path.join(self.root_dir,folder,str(self.df.iloc[idx]['image']))
        image = Image.open(img_name)
        image = PIL.ImageOps.grayscale(image)
        onehot = np.array(get_onehot(self.labellist, label))
        if self.transform:
            image = self.transform(image)
        return (image,onehot)

def getTestLoader(args):
    df_test = pd.read_csv(args['test_csv'])
    df_test = df_test.sample(frac=1)
    testdataset = XDataset(df=df_test, labellist=args['labels'], \
        root_dir=os.path.join(args['data_location'],'test'), \
        transform=transforms.Compose( \
            [transforms.Resize(args['image_dim']), \
            transforms.ToTensor(), \
            transforms.Normalize((0.5,), (0.5,))] \
            ) \
        ) 
    test_loader = torch.utils.data.DataLoader(
        testdataset,
        batch_size=args['test_batch_size'], shuffle=True)
    return test_loader

def getNumSamples(args):
    return len(pd.read_csv(args['train_csv']))

def getTrainLoader(args):
    df_train = pd.read_csv(args['train_csv'])
    # unique_class(df_train)
    df_train = df_train.sample(frac=1)
    print("-----------------")
    print(args['labels'])
    traindataset = XDataset(df=df_train, labellist=args['labels'], \
        root_dir=os.path.join(args['data_location'],'train'), \
        transform=transforms.Compose( \
            [transforms.Resize(args['image_dim']), \
            transforms.ToTensor(), \
            transforms.Normalize((0.5,), (0.5,))] \
            ) \
        )
    train_loader = torch.utils.data.DataLoader(
        traindataset,
        batch_size=args['train_batch_size'], shuffle=True)
    return train_loader