#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot
from steerDS import Net, imagePreprocessing, transform

#python scripts/deploy.py --ip 169.254.143.30

# transform = transforms.Compose(
# [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
args = parser.parse_args()

bot = PiBot(ip=args.ip)

# stop the robot 
bot.setVelocity(0, 0)

#INITIALISE NETWORK HERE

#LOAD NETWORK WEIGHTS HERE
TestPATH = f'./driveNetworks/Network_L1loss_CropThird_MoreData_2.pth'
model = Net()
model.load_state_dict(torch.load(TestPATH))
model.eval()


#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    while True:
        # get an image from the the robot
        im_cv = bot.getImage()
        im=transform(im_cv).unsqueeze(0)
        im=imagePreprocessing(im)
        # im=im[:,:,60:240,:]

        #TO DO: apply any necessary image transforms
        #print(im)
        cv2.imshow("BotCam", im_cv)
        cv2.waitKey(10)

        #TO DO: pass image through network get a prediction
        output_tensor = model(im)
        predict = output_tensor[0]
        #TO DO: convert prediction into a meaningful steering angle
        angle=predict

        #TO DO: check for stop signs?
        # angle = 0

        Kd = 10 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 15 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
