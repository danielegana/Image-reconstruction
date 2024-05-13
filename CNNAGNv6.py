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
    intdir="outputint"
    visdir="outputvis"
    argdir="outputarg"
    os.chdir(binarydir)
    directorylist=[filedir+"inputs",filedir+intdir,filedir+visdir,filedir+argdir]
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

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
    print(f"Using {device} device")
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
            feature = torch.unsqueeze(torch.tensor(image, dtype=torch.float),0).to(device)
            label = torch.tensor(params, dtype=torch.float).to(device)
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
            feature = torch.unsqueeze(torch.tensor(image, dtype=torch.float),0).to(device)
            label = torch.tensor(params, dtype=torch.float).to(device)
            return feature,label
    #%%

    class getDataset(Dataset):
        def __init__(self, root_dir,folder):
            self.root_dir = root_dir
            self.images = sorted([img for img in os.listdir(root_dir+"/"+folder) if not img.startswith('.')],key=imagenumber)
            self.params = sorted([params for params in os.listdir(root_dir+"/inputs") if not params.startswith('.')],key=imagenumber)
            self.folder=folder
        def __len__(self):
            return len(self.images)
        

        def __getitem__(self, idx):
            image_name = os.path.join(self.root_dir,self.folder ,self.images[idx])
            image = np.genfromtxt(image_name,delimiter=" ")
            param_name=os.path.join(self.root_dir,"inputs" , self.params[idx])
            params = np.genfromtxt(param_name,delimiter=" ")
            # Convert image to tensor and normalize
            feature = torch.unsqueeze(torch.tensor(image, dtype=torch.float),0).to(device)
            label = torch.tensor(params, dtype=torch.float).to(device)
            return feature,label

    class mynet(nn.Module):
# __init__ is the "standard constructor method" of the object NeuralNetwork. 
# It defines and initializes objects of the class. 
    
        def __init__(self):
            super(mynet, self).__init__()
            # 1 input image channel (black & white), 4 output channels, 5x5 square convolution
            # kernel
            npixx=32
            prepaddingy=3
            npixy=(npixx/2+1)+prepaddingy
            numparams=2
            filter1=3
            filter2=5
          #  filter3=100
            kernel1=int(5)
            kernel2=int(3)
           # kernel3=int(3)
            padding1=(kernel1-1)/2
            padding2=(kernel2-1)/2
          #  padding3=(kernel3-1)/2
          #  finalpixelx=int(((npixx-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2-(kernel3-1)+2*padding3)
          #  finalpixely=int(((npixy-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2-(kernel3-1)+2*padding3)
            finalpixelx=int(((npixx-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2)
            finalpixely=int(((npixy-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2)
            self.padding_layer = nn.ZeroPad2d((0, 0, 0, prepaddingy))
            self.conv1 = nn.Conv2d(1, filter1, kernel1, padding=int(padding1))
            self.conv2 = nn.Conv2d(filter1, filter2, kernel2,padding=int(padding2))
           # self.conv3 = nn.Conv2d(filter2, filter3, kernel3,padding=int(padding3))
            self.fc1 = nn.Linear(filter2*finalpixelx*finalpixely,10000) 
            self.fc2 = nn.Linear(10000,10000) 
            self.fc3 = nn.Linear(10000,10000) 
            self.fc4 = nn.Linear(10000,numparams) 
            #self.fc2 = nn.Linear(100, numparams)
            self.dropout = nn.Dropout(p=0.2)
            self.bn0 = nn.BatchNorm2d(1)
            self.bn1 = nn.BatchNorm2d(filter1)
            self.bn2 = nn.BatchNorm2d(filter2)
           # self.bn3 = nn.BatchNorm2d(filter3)
            self.bnf1 = nn.BatchNorm1d(10000)
            self.bnf2 = nn.BatchNorm1d(numparams)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x=F.pad(x,(0,3),"constant",0)
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(self.bn0(x)))), (2, 2))
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
          #  x = F.relu(self.bn3(self.conv3(x)))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.bnf1(self.fc1(x)))
        #   x= self.dropout(x)
            x = F.relu(self.bnf1(self.fc2(x)))
         #   x = F.relu(self.bnf1(self.fc3(x)))
            x = F.relu(self.bnf2(self.fc4(x)))

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
            if batch % 10 == 0:
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
    
    # %%
  #  agndatavis=visDataset(filedir)
  #  agndataint=ImageDataset(filedir)
    agndatavis=getDataset(filedir,folder=visdir)
    agndataint=getDataset(filedir,folder=intdir)
    agndataarg=getDataset(filedir,folder=argdir)
    visflag=1

#%%
    if visflag==1:
        agndata=agndatavis
    else:
        agndata=agndataint

    train_size = int(0.8 * len(agndata))
    val_size = len(agndata) - train_size

    train_dataset = Subset(agndata, range(0,train_size))
    test_dataset = Subset(agndata, range(train_size,len(agndata)))

    test_images,test_labels=[x for x, y in test_dataset], [y for x, y in test_dataset]
    test_images=torch.stack([matrix for matrix in  test_images]).to(device)
    test_labels=torch.stack([matrix for matrix in  test_labels]).to(device)

   
    #%% CNN training
    model = mynet().to(device)
    batch_size=200
    epochs=50
    learningrate=1e-2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), learningrate,momentum=0.9)    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)    
        print("Done!")

  # %% Compute the mse for the test images
    y_pred = model(test_images)
    frac_error=model(test_images)/test_labels
    mse = F.mse_loss(test_labels, y_pred)
    print("MSE CNN: ",mse)
    #print("Target labels: ",test_labels)
    print("Fractional Error: ",frac_error)

    #%% Benchmark is the average inclination

    y_pred =  torch.tensor([0.5,0.5]).unsqueeze(0).repeat(len(test_labels),1).to(device)
    mse = F.mse_loss(test_labels, y_pred)
    print("MSE Benchmark: ",mse)



#################################################################
    #%%
    #%% Data preparation for DT
    train_imagesDT,train_labelsDT=[torch.squeeze(x.view(x.size(1)*x.size(2), -1),1) for x, y in train_dataset], [y for x, y in train_dataset]
    train_imagesDT=torch.stack([matrix for matrix in  train_imagesDT]).cpu()
    train_labelsDT=torch.stack([matrix for matrix in  train_labelsDT]).cpu()

    #train_imagesDT=np.stack(train_imagesDT)
    #train_labelsDT=np.stack(train_labelsDT)

    #%%
    test_imagesDT,test_labelsDT=[torch.squeeze(x.view(x.size(1)*x.size(2), -1),1) for x, y in test_dataset], [y for x, y in test_dataset]
    test_imagesDT=torch.stack([matrix for matrix in  test_imagesDT]).cpu()
    test_labelsDT=torch.stack([matrix for matrix in  test_labelsDT]).cpu()
    #test_imagesDT,test_labelsDT=[torch.squeeze(x.view(x.size(1)*x.size(2), -1),1).numpy() for x, y in test_dataset], [y for x, y in test_dataset]
    #test_imagesDT=np.stack(test_imagesDT)
    #test_labelsDT=np.stack(test_labelsDT)

    ### A DT
    # %%
    modelDT=ensemble.RandomForestRegressor()
    modelDT.fit(train_imagesDT,train_labelsDT)

 
    # %% Now compute the mse for the test images
    y_pred = modelDT.predict(test_imagesDT)
    frac_error=np.divide(modelDT.predict(test_imagesDT),test_labelsDT)
    mse = mean_squared_error(test_labelsDT, y_pred)
    print("MSE DT: ",mse)
    #print("Target labels: ",test_labels)
    print("Fractional Error: ",frac_error)

    # %%

    ### PLOTS
    # %%
    plt.figure(figsize=(20, 15))
    plt.imshow(agndatavis[0][0][0].cpu()[0:10,0:10])
    plt.figure(figsize=(20, 15))
    plt.imshow(agndataint[0][0])

    # %%
    plt.imshow(agndataint[0][0])
    # %%
    plt.figure(figsize=(20, 15))
    plt.imshow(agndata[0][0][0:10,0:10])

    # %%
