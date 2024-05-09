#%%

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset,random_split
import matplotlib.pyplot as plt
from numpy import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree
from sklearn import ensemble

import torch.nn.init as init
import re
import os
from sklearn.metrics import mean_squared_error
import itertools
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from itertools import starmap
import subprocess
from helpers import plot
from itertools import chain
import bashpool

#%%


# Test func pool function
if __name__ == '__main__':

    pythondir="/Users/danielegana/Dropbox (PI)/ML/code/AGN/"
    filedir="/Users/danielegana/Dropbox (PI)/ML/code/agndisk/agndisk/files/"
    binarydir="/Users/danielegana/Dropbox (PI)/ML/code/agndisk/agndisk/"
    os.chdir(pythondir)
    print("lala")
    #%%
    lendata=1000
    inclination = np.random.uniform(0, np.pi/2, lendata)
    R0 = np.random.uniform(6, 100, lendata)

    paramtuples = [tuple(row) for row in np.column_stack((inclination, R0))]

    random.shuffle(paramtuples)
    #%%

    os.chdir(binarydir)
    directorylist=[filedir+"inputs",filedir+"outputint",filedir+"outputvis"]
    #%%
    for x in directorylist:
        os.makedirs(x, exist_ok=True)
        files = os.listdir(x)
        for file in files:
            file_path = os.path.join(x, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    #%%
    #num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    #with multiprocessing.Pool(processes=num_processes) as pool:
    #    printout = pool.starmap(bashpool.testfun, paramtuples)
    #%%
    for x,(inclinationit,R0it) in enumerate(paramtuples):
        with open(filedir+"inputs/input" + str(x), 'w') as file:
        # Write each number to the file
            file.write(str(inclinationit/np.max(inclination))+" "+str(R0it/np.max(R0)) + '\n')
    
    #%%
    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        printout = pool.starmap(bashpool.bashfun,list(enumerate(paramtuples)))
    #%%
    def imagenumber(imagefile):
        return int(re.findall(r'\d+', imagefile)[0])

    class ImageDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.images = sorted([img for img in os.listdir(root_dir+"/outputint") if not img.startswith('.')],key=imagenumber)
            self.params = sorted([params for params in os.listdir(root_dir+"/inputs") if not params.startswith('.')],key=imagenumber)
        
        def __len__(self):
            return len(self.images)
        

        def __getitem__(self, idx):
            image_name = os.path.join(self.root_dir,"outputint" ,self.images[idx])
            image = np.genfromtxt(image_name,delimiter=" ")
            param_name=os.path.join(self.root_dir,"inputs" , self.params[idx])
            params = np.genfromtxt(param_name,delimiter=" ")
            # Convert image to tensor and normalize
            feature = torch.tensor(image, dtype=torch.float) 
            label = torch.tensor(params, dtype=torch.float)
            return feature,label
        
    class visDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.images = sorted([img for img in os.listdir(root_dir+"/outputvis") if not img.startswith('.')],key=imagenumber)
            self.params = sorted([params for params in os.listdir(root_dir+"/inputs") if not params.startswith('.')],key=imagenumber)
        
        def __len__(self):
            return len(self.images)
        

        def __getitem__(self, idx):
            image_name = os.path.join(self.root_dir,"outputvis" ,self.images[idx])
            image = np.genfromtxt(image_name,delimiter=" ")
            param_name=os.path.join(self.root_dir,"inputs" , self.params[idx])
            params = np.genfromtxt(param_name,delimiter=" ")
            # Convert image to tensor and normalize
            feature = torch.tensor(image, dtype=torch.float) 
            label = torch.tensor(params, dtype=torch.float)
            return feature,label
    #%%


    class mynet(nn.Module):
# __init__ is the "standard constructor method" of the object NeuralNetwork. 
# It defines and initializes objects of the class. 
    
        def __init__(self):
            super(mynet, self).__init__()
            # 1 input image channel (black & white), 4 output channels, 5x5 square convolution
            # kernel
            filter1=50
            filter2=100
            filter3=200
            kernel1=int(50)
            kernel2=int(30)
            kernel3=int(10)
            padding1=(kernel1-1)/2
            padding2=(kernel2-1)/2
            padding3=(kernel3-1)/2
            finalpixel=int(((28-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2-(kernel3-1)+2*padding3)
            self.conv1 = nn.Conv2d(1, filter1, kernel1, padding=int(padding1))
            self.conv2 = nn.Conv2d(filter1, filter2, kernel2,padding=int(padding2))
            self.conv3 = nn.Conv2d(filter2, filter3, kernel3,padding=int(padding3))
            self.fc1 = nn.Linear(filter3*finalpixel*finalpixel,100)  # 5*5 from image dimension
            self.fc2 = nn.Linear(100, 10)
            self.dropout = nn.Dropout(p=0.2)
            self.bn1 = nn.BatchNorm2d(filter1)
            self.bn2 = nn.BatchNorm2d(filter2)
            self.bn3 = nn.BatchNorm2d(filter3)
            self.bnf1 = nn.BatchNorm1d(100)
            self.bnf2 = nn.BatchNorm1d(10)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
            x=self.dropout(x)
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
            x = F.relu(self.bn3(self.conv3(x)))
        # x=self.dropout(x)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.bnf1(self.fc1(x)))
        #  x= self.dropout(x)
            x = F.relu(self.bnf2(self.fc2(x)))
            return x
        
        def num_flat_features(self, x):
            size = x.size()[1:] # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features
    
  #%%
    #Defines the function to train the model
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        #Initializes the training mode of the model
        model.train()
        #enumerate creates a tuple index,data. so batch gets the index number
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            # The loss is computed against the known class label. y is an integer, pred is a 10-dimensional vector
            # with the 10 classes. 
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            #optimizer.zero_grad() zeroes out the gradient after one pass. this is to 
            #avoid accumulating gradients, which is the standard behavior
            optimizer.zero_grad()

            # Print loss every 100 batches
            if batch % 100 == 0:
                loss, current = loss.item(), batch*batch_size
                print(f"loss: {loss:>7f}  [{current:>5d}]")

# %%
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct







    agndatavis=visDataset(filedir)
    agndataint=ImageDataset(filedir)
    visflag=1

    if visflag==1:
        agndata=agndatavis
    else:
        agndata=agndataint


    train_size = int(0.8 * len(agndata))
    val_size = len(agndata) - train_size

    #train_dataset, test_dataset = random_split(agndata, [train_size, val_size])
    train_dataset = Subset(agndata, range(0,train_size))
    test_dataset = Subset(agndata, range(train_size,len(agndata)))


    batch_size=20
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)


    #%%
    train_images,train_labels=[torch.squeeze(x.view(x.size(0)*x.size(1), -1),1).numpy() for x, y in train_dataset], [y for x, y in train_dataset]
    train_images=np.stack(train_images)
    train_labels=np.stack(train_labels)

    #%%

    test_images,test_labels=[torch.squeeze(x.view(x.size(0)*x.size(1), -1),1).numpy() for x, y in test_dataset], [y for x, y in test_dataset]
    test_images=np.stack(test_images)
    test_labels=np.stack(test_labels)

    ### A DT
    # %%
    model=ensemble.RandomForestRegressor()
    model.fit(train_images,train_labels)

    #%% Benchmark is the average inclination

    y_pred = np.tile([0.5,0.5], (len(test_labels), 1))
    mse = mean_squared_error(test_labels, y_pred)
    print("MSE Benchmark: ",mse)


    # %% Now compute the mse for the test images
    y_pred = model.predict(test_images)
    frac_error=np.divide(model.predict(test_images),test_labels)
    mse = mean_squared_error(test_labels, y_pred)
    print("MSE DT: ",mse)
    #print("Target labels: ",test_labels)
    print("Fractional Error: ",frac_error)

    # %%

    ### PLOTS
    # %%
    plt.figure(figsize=(20, 15))
    plt.imshow(agndatavis[0][0][0:10,0:10])
    plt.figure(figsize=(20, 15))
    plt.imshow(agndataint[0][0])

    # %%
    plt.imshow(agndataint[0][0])
    # %%
    plt.figure(figsize=(20, 15))
    plt.imshow(agndata[0][0][0:10,0:10])

    # %%
