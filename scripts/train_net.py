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
from steerDS import SteerDataSet, Net, Old_Net, ConvMixer

'''DAY1 Data COllection'''
day1Filenames=['TrackShort_Kd=10', 'TrackShort_Kd=15', 'Track0_Kd=5_Ka=15','Track0_Kd=10_Ka=25', 'Track0_Kd=20_Ka=25', 'Track1_Kd=5_Ka=15', 'TrackMed_Kd=10', 'TrackLong']
# trainingFolderName=day1Filenames[7]
testingFolderName=day1Filenames[7] #[5]

'''DAY2 Morning Data Collection'''
day2Filenames=['TrackLong_Kd=10']
# trainingFolderName=day2Filenames[0]
testingFolderName=day2Filenames[0]

'''DAY4 Morning Data Collection'''
day4Filenames=['TrackLongest_Kd=10_Ka=10','TrackSegments', 'EvenDistribution']
# day4Filenames=['EvenDistribution','TrackSegments', 'EvenDistribution']
trainingFolderName=day4Filenames[2]
testingFolderName=day4Filenames[0]

'''TrainedNetworks'''
folderName='driveNetworks/'
# TestPATH = f'./{folderName}Network_L1loss_CropThird_ConvMixer.pth'
# TestPATH = f'./{folderName}Network_L1loss_CropThird_MoreData.pth' #47 but more consistent
# TestPATH = f'./{folderName}Network_L1loss_CropThird_MoreData_2.pth'  #45
TestPATH = f'./{folderName}Network_L1loss_CropThird_Segments_Even1.pth'   #
# TestPATH = f'./{folderName}Network_L1loss_CropThird_Segments_SGD.pth' #40.6
# TestPATH = f'./{folderName}Network_L1Loss_PrevAngl.pth'           #34
# TestPATH = f'./{folderName}Network_MSEloss_CropThird.pth'
# TestPATH = f'./{folderName}Network_L1loss_CropHalf_Normalise.pth' #48
# TestPATH = f'./{folderName}Network_L1loss_CropHalf.pth'           #53
# TestPATH = f'./{folderName}Network_L1loss_CropThird.pth'          #62
# TestPATH = f'./{folderName}Network_L1loss_CropThird_1.pth'        #41
# TestPATH = f'./{folderName}Train_Track1_Kd=10.pth'
# TestPATH = f'./{folderName}Train_track2.pth'

transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((100,100))]
    )

script_path = os.path.dirname(os.path.realpath(__file__))

ds_train = SteerDataSet(os.path.join(script_path, '..', 'data', 'train', trainingFolderName), '.jpg', transform)
ds_train_dataloader = DataLoader(ds_train,batch_size=8,shuffle=True)

ds_test = SteerDataSet(os.path.join(script_path, '..', 'data', 'train', testingFolderName), '.jpg', transform)
ds_test_dataloader = DataLoader(ds_test,batch_size=1,shuffle=True)

def analyseData():
    fig, (ax1,ax2)= plt.subplots(1,2)
    all_y = []
    for S in ds_train_dataloader:
        im, y = S
        all_y += y.tolist()

    #     ax1.imshow(im.squeeze(0).permute(1, 2, 0).numpy())
    #     im=im[:,:,60:240,:]  
    #     ax2.imshow(im.squeeze(0).permute(1, 2, 0).numpy())
    #     plt.pause(0.1)
    # plt.show()
    plt.hist(all_y, bins = 100)
    plt.show()
    print(f"The train dataset contains {len(ds_train)} images and test dataset contain {len(ds_test)} images ")
    print(f'Input shape: {im.shape}')
    print('Outputs and their counts:')
    print(np.unique(all_y, return_counts = True))


def imagePreprocessing(im, flip=1):
    
    im=im[:,:,im.size(2)//3:,:]  #third 
    
    # if flip==-1:
    #     im=torch.flip(im, (3,))
    # im=im[:,:,120:240,:]  
    # im=F.local_response_norm(im, size=5)
    return im


def training(numEpochs=10, net=Net()):
    # net=Net()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
    steerFlip=[1,-1]
    for epoch in range(numEpochs):  # loop over the dataset multiple times
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(ds_train_dataloader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # flip image and label
            # labels=labels
            inputs=imagePreprocessing(inputs)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
        print('')
        # print(f'Avergae loss: {running_loss/len(ds_train)}')
        # end for over minibatches epoch finishes
        end_time = time.time()
        # plt.show()
        

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
                labels=labels
                images=imagePreprocessing(images)
                output = net(images)
                
                # print(output)
                # note here we take the max of all probability
                predicted = output[0]#torch.max(output, 1)
                # print(output, labels)
                
                predLables.append(predicted)
                GT.append(labels)
                total += labels.size(0)
                correct += (abs(predicted - labels)).sum().item()
            
        print('Epoch', epoch+1, 'took', end_time-start_time, 'seconds')
        print('Error of the network after', epoch+1, 'epochs is' , correct/total)
        
    print('Finished Training')
    torch.save(net.state_dict(), TestPATH)


def testing(TestPATH, model=Net(), plot=True):
    # model = Net()
    model.load_state_dict(torch.load(TestPATH))
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
            images=imagePreprocessing(images)

            output_tensor = model(images)
            # note here we take the max of all probability
            predicted = output_tensor[0]#torch.max(output, 1)
            # print(output)
            
            predLables.append(predicted.detach().numpy())
            GT.append(labels.detach().numpy())
            total += labels.size(0)
            correct += (abs(predicted - labels)<0.1).sum().item()
    
    print('Accuracy of the network after is' , 100*correct/total)
    if plot==True:
        fig, (ax1,ax2)= plt.subplots(1,2)
        ax1.plot(np.arange(total), GT, 'g.-')
        ax1.plot(np.arange(total), predLables, 'm.-')

        ax2.hist(abs(np.array(GT)-np.array(predLables)), bins=50)
        ax2.set_xlim([0,3.14])

        plt.show()


# analyseData()
training(numEpochs=100)

testing(TestPATH)

# Training(numEpochs=10, net=ConvMixer(124,8))
# Testing(TestPATH,  model=ConvMixer())

'''Loop through test data'''
# for i in range(7):
#     testingFolderName=day1Filenames[i] #[5]
#     ds_test = SteerDataSet(os.path.join(script_path, '..', 'data', 'train', testingFolderName), '.jpg', transform)
#     ds_test_dataloader = DataLoader(ds_test,batch_size=1,shuffle=True)
#     Testing(TestPATH,  model=Net(), plot=False)
