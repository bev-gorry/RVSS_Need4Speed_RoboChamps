import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 
import matplotlib.pyplot as plt
from steerDS import SteerDataSet


trainingFolderName='Track1_Kd=10'
testingFolderName='Track1_Kd=15'

trainingFolderName='track2'
# testingFolderName='track1'
testingFolderName='Track1_Kd=10'

transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

script_path = os.path.dirname(os.path.realpath(__file__))


ds_train = SteerDataSet(os.path.join(script_path, '..', 'data', 'train', trainingFolderName), '.jpg', transform)
ds_train_dataloader = DataLoader(ds_train,batch_size=1,shuffle=True)

ds_test = SteerDataSet(os.path.join(script_path, '..', 'data', 'train', testingFolderName), '.jpg', transform)
ds_test_dataloader = DataLoader(ds_test,batch_size=1,shuffle=True)


all_y = []
for S in ds_train_dataloader:
    im, y = S    
    all_y += y.tolist()
print(f"The train dataset contains {len(ds_train)} images and test dataset contain {len(ds_test)} images ")
print(f'Input shape: {im.shape}')
print('Outputs and their counts:')
print(np.unique(all_y, return_counts = True))

class Net(nn.Module):
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


def Training(numEpochs=10):
    net=Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(numEpochs):  # loop over the dataset multiple times
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(ds_train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # if i % 20 == 19:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            #     running_loss = 0.0
        print('')
        print(f'Avergae loss: {running_loss/len(ds_train)}')
        # end for over minibatches epoch finishes
        end_time = time.time()

        # test the network every epoch on test example
        correct = 0
        total = 0
        predLables,GT=[],[]
        # Test after the epoch finishes (no gradient computation needed)
        with torch.no_grad():
            VIS = True
            for data in ds_test_dataloader:
                # load images and labels
                images, labels = data

                output = net(images)
                # note here we take the max of all probability
                predicted = output[0]#torch.max(output, 1)
                # print(output)
                
                predLables.append(predicted)
                GT.append(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Epoch', epoch+1, 'took', end_time-start_time, 'seconds')
        print('Accuracy of the network after', epoch+1, 'epochs is' , 100*correct/total)
        
    print('Finished Training')
    PATH = f'./Train_{trainingFolderName}.pth'
    torch.save(net.state_dict(), PATH)

    plt.plot(np.arange(total), GT, 'g.-')
    plt.plot(np.arange(total), predLables, 'm.-')

    plt.show()

def Testing():
    PATH = f'./Train_Track1_Kd=10.pth'
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # test the network every epoch on test example
    correct = 0
    total = 0
    predLables,GT=[],[]
    # Test after the epoch finishes (no gradient computation needed)
    with torch.no_grad():
        VIS = True
        for data in ds_test_dataloader:
            # load images and labels
            images, labels = data

            output_tensor = model(images)
            # note here we take the max of all probability
            predicted = output_tensor[0]#torch.max(output, 1)
            # print(output)
            
            predLables.append(predicted)
            GT.append(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network after is' , 100*correct/total)

    plt.plot(np.arange(total), GT, 'g.-')
    plt.plot(np.arange(total), predLables, 'm.-')

    plt.show()


Testing()

