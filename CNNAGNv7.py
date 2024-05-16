#%%

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from torch.utils.data import Dataset,random_split
import matplotlib.pyplot as plt
from numpy import random
import torch.nn.functional as F
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
from itertools import chain
import bashpoolfull


#%%

# Test func pool function
if __name__ == '__main__':

    clusterrun="true"

    if clusterrun == "false":
        pythondir="/Users/danielegana/Dropbox (PI)/ML/code/AGN/"
        filedir="/Users/danielegana/files/"
        binarydir="/Users/danielegana/Dropbox (PI)/ML/code/agndisk/agndisk/"
        from helpers import plot


    if clusterrun == "true":
        pythondir="/home/degana/agn/AGN"
        filedir="/home/degana/agn/files/"
        binarydir="/home/degana/agn"

    os.chdir(pythondir)
    #%%
    lendata=100
    numpix=int(32)
    numpixarray=[numpix]*lendata
    numparamvis=int(numpix*(numpix/2+1))
    Rg= np.random.uniform(1, 10, lendata)
    Rin=np.random.uniform(Rg, 100, lendata)
    R0 = np.random.uniform(Rin, 100, lendata)
    slope = np.random.uniform(1, 5, lendata)
    Rmax=150
   # inclination = np.cos(np.random.uniform(0, np.pi/2, lendata))
   # posangle = np.cos(np.random.uniform(0, 2*np.pi, lendata))
    inclination = np.random.uniform(0.01, 1, lendata)
    posangle = np.random.uniform(-1, 1, lendata)

    #%%
    paramtuples = [tuple(row) for row in np.column_stack((Rg, Rin, R0, slope, inclination, posangle, numpixarray))]
    selecttuples=[0,1,2,4,5]
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
    print("Create input files:")
        
    for x,(Rgit, Rinit,R0it, slopeit, inclinationit, posangleit, numpixit), in enumerate(paramtuples):
        with open(filedir+"inputs/input" + str(x), 'w') as file:
        # Write each number to input files for net training. 
        # Writing order is: Rg, Rin, R0, slope, inclination, posangle. This is for the C runner and label retrieving. In bashfun I take the arccos of angles to send to C runner. /np.max(inclination) /np.max(posangle)
            strintemp=""
            if bashpoolfull.find_number(selecttuples, 0):
                strintemp=strintemp+" "+str(Rgit/np.max(Rg))
            if bashpoolfull.find_number(selecttuples, 1):
                strintemp=strintemp+" "+str(Rinit/np.max(Rin))
            if bashpoolfull.find_number(selecttuples, 2):
                strintemp=strintemp+" "+str(R0it/np.max(R0))
            if bashpoolfull.find_number(selecttuples, 3):
                strintemp=strintemp+" "+str(slopeit/np.max(slope))
            if bashpoolfull.find_number(selecttuples, 4):
                strintemp=strintemp+" "+str(inclinationit)
            if bashpoolfull.find_number(selecttuples, 5):
                strintemp=strintemp+" "+str(posangleit)
            file.write(strintemp + '\n')
          #  file.write(str(Rgit/np.max(Rg))+" "+str(Rinit/np.max(Rin))+" "+str(R0it/np.max(R0))+" "+str(slopeit/np.max(slope))+" "+str(inclinationit)+" "+str(posangleit) + '\n')
         #   file.write(str(Rgit/Rmax)+" "+str(Rinit/Rmax)+" "+str(R0it/Rmax)+" "+str(slopeit)+" "+str(inclinationit)+" "+str(posangleit) + '\n')
    #
    tuplelist=list(enumerate(paramtuples))
    tuplelist= [(tup[0], tup[1], selecttuples) for tup in tuplelist]
    #%% Run C pool
    print("Start C pool")

    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        printout = pool.starmap(bashpoolfull.bashfun,tuplelist)
    #%%
    def imagenumber(imagefile):
        return int(re.findall(r'\d+', imagefile)[0])

    #testimage=np.transpose(np.genfromtxt("/Users/danielegana/files/outputint/intensityfile"))
    #plt.imshow(testimage)

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
    print(f"Finished C pool. Using {device} device")
 
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
        
    class getDatasetphasereco(Dataset):
        def __init__(self, root_dir,foldervis,folderarg):
            self.root_dir = root_dir
            self.vis = sorted([img for img in os.listdir(root_dir+"/"+foldervis) if not img.startswith('.')],key=imagenumber)
            self.arg = sorted([img for img in os.listdir(root_dir+"/"+folderarg) if not img.startswith('.')],key=imagenumber)
            self.foldervis=foldervis
            self.folderarg=folderarg

        def __len__(self):
            return len(self.vis)
        

        def __getitem__(self, idx):
            vis_name = os.path.join(self.root_dir,self.foldervis ,self.vis[idx])
            vis = np.genfromtxt(vis_name,delimiter=" ")

            arg_name = os.path.join(self.root_dir,self.folderarg ,self.arg[idx])
            arg = np.genfromtxt(arg_name,delimiter=" ")
           
            # Convert image to tensor and normalize
            feature = torch.unsqueeze(torch.tensor(vis, dtype=torch.float),0).to(device) #Unsqueezing needed to create a dummy 1-filter ("black and white")
            label = torch.tensor(arg, dtype=torch.float).to(device) #No unsqueezing needed for the output
            return feature,label

    class mynet(nn.Module):
# __init__ is the "standard constructor method" of the object NeuralNetwork. 
# It defines and initializes objects of the class. 
    
        def __init__(self,npixx,numparams,phasereco):
            super(mynet, self).__init__()
            # 1 input image channel (black & white), 4 output channels, 5x5 square convolution
            # kernel
            self.phasereco=phasereco
            prepaddingy=3
            npixy=(npixx/2+1)+prepaddingy
            self.npixx=int(npixx)
            self.npixy=int(npixx/2+1)
            filter1=10
            filter2=10
            kernel1=int(5)
            kernel2=int(3)
           # kernel3=int(3)
            padding1=(kernel1-1)/2
            padding2=(kernel2-1)/2
           # padding3=(kernel3-1)/2
            layer1size=100
            layer2size=100
            layer3size=100
            finalpixelx=int(((npixx-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2)
            finalpixely=int(((npixy-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2)
           # finalpixelx=int(((npixx-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2-(kernel3-1)+2*padding3)
           # finalpixely=int(((npixy-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2-(kernel3-1)+2*padding3)
            self.padding_layer = nn.ZeroPad2d((0, 0, 0, prepaddingy))
            self.conv1 = nn.Conv2d(1, filter1, kernel1, padding=int(padding1))
            self.conv2 = nn.Conv2d(filter1, filter2, kernel2,padding=int(padding2))
            self.fc1 = nn.Linear(filter2*finalpixelx*finalpixely,layer1size) 
            self.fc2 = nn.Linear(layer1size,layer2size) 
            self.fc3 = nn.Linear(layer2size,layer3size) 
            self.fc4 = nn.Linear(layer3size,numparams) 
            self.dropout = nn.Dropout(p=0.1)
            self.bn0 = nn.BatchNorm2d(1)
            self.bn1 = nn.BatchNorm2d(filter1)
            self.bn2 = nn.BatchNorm2d(filter2)
            self.bnf1 = nn.BatchNorm1d(layer1size)
            self.bnf2 = nn.BatchNorm1d(layer2size)
            self.bnf3 = nn.BatchNorm1d(layer3size)


        def forward(self, x):
            # Max pooling over a (2, 2) window
            x=F.pad(x,(0,3),"constant",0)
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
          #  x = F.relu(self.bn3(self.conv3(x)))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.bnf1(self.fc1(x)))
            x= self.dropout(x)
            x = F.relu(self.bnf2(self.fc2(x)))
            x= self.dropout(x)
            x = F.relu(self.bnf3(self.fc3(x)))
            x= self.dropout(x)
            x = self.fc4(x)
            if self.phasereco==False:
              #  return torch.sigmoid(x) #return the sigmoid of x to regularize output btn 0 and 1
                return x
            if self.phasereco==True:
                x=x.reshape(x.shape[0],self.npixx,self.npixy)
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
    # READ DATASETS
    agndatavis=getDataset(filedir,folder=visdir)
    agndataint=getDataset(filedir,folder=intdir)
    visflag=1
    agndataphasereco=getDatasetphasereco(filedir,foldervis=visdir,folderarg=argdir)

#################################################################
##### Dataset import
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

    #%%
  #  model = mynet(npixx=numpix,numparams=(len(paramtuples[1])-1),phasereco=False).to(device)


#################################################################
    selectalgorithm="DT" # "CNN" or "CNNPHASE" or "DT"
#################################################################
##### CNN PARAMETER REGRESSION
    if selectalgorithm=="CNN":
    #%% CNN training
        model = mynet(npixx=numpix,numparams=(len(selecttuples)),phasereco=False).to(device)
        batch_size=100
        epochs=30
        learningrate=1e-3
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), learningrate)    
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)    
            print("Done!")

    # %% Compute the mse for the test images
        y_pred = model(test_images)
        frac_error=model(test_images)/test_labels
        mse = F.mse_loss(test_labels, y_pred)
        mae = mse.sqrt()
        print("MFRE per feature",np.mean(torch.detach(frac_error.cpu()).numpy(),axis=0))
        print("MAE CNN: ",mae)
        print("MSE CNN: ",mse)
        #print("Target labels: ",test_labels)
        print("Fractional Error: ",frac_error)

        #%% Benchmark is the average inclination

        y_pred =  torch.tensor([0.5,0.5]).unsqueeze(0).repeat(len(test_labels),1).to(device)
        mse = F.mse_loss(test_labels, y_pred)
        print("MSE Benchmark: ",mse)

#################################################################
    #%% 
    #CNN FOR FULL PHASE RECONSTRUCTION
    # Train for pure phase reconstruction

    if selectalgorithm=="CNNphase":

        train_size = int(0.8 * len(agndataphasereco))
        val_size = len(agndataphasereco) - train_size

        train_dataset = Subset(agndataphasereco, range(0,train_size))
        test_dataset = Subset(agndataphasereco, range(train_size,len(agndataphasereco)))

        test_images,test_labels=[x for x, y in test_dataset], [y for x, y in test_dataset]
        test_images=torch.stack([matrix for matrix in  test_images]).to(device)
        test_labels=torch.stack([matrix for matrix in  test_labels]).to(device)
        #%%
        model = mynet(npixx=numpix,numparams=numparamvis,phasereco=True).to(device)

        #%% CNN training
        model = mynet(npixx=numpix,numparams=numparamvis,phasereco=True).to(device)
        model.train()
        batch_size=100
        epochs=10
        learningrate=1e-4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), learningrate,momentum=0.9)    
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)    
            print("Done!")

        #%%
        y_pred = model(test_images)
        frac_error=model(test_images)/test_labels
        mse = F.mse_loss(test_labels, y_pred)
        print("MSE CNN: ",mse)
        #print("Target labels: ",test_labels)
        print("Fractional Error: ",frac_error)

        #model(agndatavis[0][0].unsqueeze(0))[0].cpu().detach().numpy()
        #%%
        # Getting the image from the phase and the visibility
        
        def getimage(agndatavis,idx,npix):
            model.eval()
            phase=model(agndatavis[idx][0].unsqueeze(0))[0].cpu().detach().numpy()
            amp=np.sqrt(agndatavis[idx][0][0].cpu().numpy())
            complex=amp*np.exp(1j * phase)
            return np.fft.irfft2(complex)
        
        def getdummyimage(agndatavis,idx,npix):
            model.eval()
            amp=np.sqrt(agndatavis[idx][0][0].cpu().numpy())
            return np.fft.irfft2(amp)
        
        #%% Test image reco
        plt.subplot(1, 3, 1)
        plt.imshow(agndataint[0][0][0].cpu())
        plt.title("Original")
        plt.subplot(1, 3, 2)
        plt.imshow(getdummyimage(agndatavis,0,numpix))
        plt.title("Dummy zero phase")
        plt.subplot(1, 3, 3)
        plt.imshow(getimage(agndatavis,0,numpix))
        plt.title("Reconstructed")



#################################################################
    #%%
    #%% Data preparation for DT

    if selectalgorithm=="DT":
    
        train_imagesDT,train_labelsDT=[torch.squeeze(x.view(x.size(1)*x.size(2), -1),1) for x, y in train_dataset], [y for x, y in train_dataset]
        train_imagesDT=torch.stack([matrix for matrix in  train_imagesDT]).cpu().numpy()
        train_labelsDT=torch.stack([matrix for matrix in  train_labelsDT]).cpu().numpy()

        #train_imagesDT=np.stack(train_imagesDT)
        #train_labelsDT=np.stack(train_labelsDT)

        #%%
        test_imagesDT,test_labelsDT=[torch.squeeze(x.view(x.size(1)*x.size(2), -1),1) for x, y in test_dataset], [y for x, y in test_dataset]
        test_imagesDT=torch.stack([matrix for matrix in  test_imagesDT]).cpu().numpy()
        test_labelsDT=torch.stack([matrix for matrix in  test_labelsDT]).cpu().numpy()
        #test_imagesDT,test_labelsDT=[torch.squeeze(x.view(x.size(1)*x.size(2), -1),1).numpy() for x, y in test_dataset], [y for x, y in test_dataset]
        #test_imagesDT=np.stack(test_imagesDT)
        #test_labelsDT=np.stack(test_labelsDT)

        ### A DT
        # %%
        modelDT=ensemble.RandomForestRegressor()
        #modelDT=ensemble.BaggingRegressor()
    # modelDT=tree.DecisionTreeRegressor(min_samples_split=10,min_samples_leaf=10)


        modelDT.fit(train_imagesDT,train_labelsDT)

    
        # %% Now compute the mse for the test images
        y_pred = modelDT.predict(test_imagesDT)
        frac_error=np.divide(modelDT.predict(test_imagesDT),test_labelsDT)
        mse = mean_squared_error(test_labelsDT, y_pred)
        mae = np.sqrt(mean_squared_error(test_labelsDT, y_pred))
        print("MFRE per feature",np.mean(frac_error, axis=0))
        print("MSE DT: ",mse)
        print("MAE DT: ",mae)

        #print("Target labels: ",test_labels)
        print("Fractional Error: ",frac_error)


