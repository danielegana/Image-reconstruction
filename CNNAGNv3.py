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
    filedir="/Users/danielegana/Dropbox (PI)/ML/code/AGN/agndisk/agndisk/files/"
    binarydir="/Users/danielegana/Dropbox (PI)/ML/code/AGN/agndisk/agndisk/"
    os.chdir(pythondir)
    print("lala")
    #%%
    lendata=100
    inclination = np.random.uniform(0.1, 0.1, lendata)
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


    #batch_size=20
    #train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)


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
    model=tree.DecisionTreeRegressor()
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
