import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path
import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((100,100))]
    )

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
       
        f=self.filenames[idx]
        img = cv2.imread(f)
        
        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        # steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        # steering = np.float32(steering)    
        if f[-9]== '-':
            steer=f[-9:][:-4]
        else: 
            steer=f[-8:][:-4]    
        steering = np.float32(steer)               
        return img, steering

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 32, 5)
        # self.conv4 = nn.Conv2d(32, 64, 5)
        # self.conv3 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(4576,520)
        self.fc1 = nn.Linear(51744,120)
        # self.fc2 = nn.Linear(520, 84)
        self.fc3 = nn.Linear(120, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.size())
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x=(x-0.5)*2*3.14
        return x



# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.conv3 = nn.Conv2d(16, 32, 5)
#         self.conv4 = nn.Conv2d(32, 64, 5)
#         # self.conv3 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(4576,520)
#         # self.fc1 = nn.Linear(33264,120)
#         # self.fc2 = nn.Linear(520, 84)
#         self.fc3 = nn.Linear(520, 1)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))

#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         # print(x.size())
#         x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         x=(x-0.5)*2
#         return x


class Old_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(3744,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x=(x-0.5)*2*3.14
        return x
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x=torch.sigmoid(self.fn(x))
        x=(x-0.5)*2*3.14
        return x
        # (torch.sigmoid(nn.Linear(dim, nclasses))-0.5)*2*3.14
        # return self.fn(x) + x
    

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, nclasses=1):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, nclasses)
        )


def imagePreprocessing(im, flip=1):
    
    im=im[:,:,im.size(2)//3:,:]  #third 
    
    # if flip==-1:
    #     im=torch.flip(im, (3,))
    # im=im[:,:,120:240,:]  
    # im=F.local_response_norm(im, size=5)
    return im
