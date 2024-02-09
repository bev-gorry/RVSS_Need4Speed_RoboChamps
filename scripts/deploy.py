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
from torchvision import transforms
from machinevisiontoolbox import Image 
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
TestPATH = f'./driveNetworks/Network_L1loss_CropThird_Segments_Even0.pth'
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
        # im=transform(im_cv).numpy()
        # im=im[:,:,np.shape(im)[1]//3:]
        # im=np.swapaxes(im,0,2)
        # im=np.swapaxes(im,0,1)
        # print(np.shape(im))
        
        hsv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([118, 86, 51])
        upper_red1 = np.array([179, 211, 239])
        lower_red2 = np.array([0, 86, 51])
        upper_red2 = np.array([3, 211, 239])

        # lower_red = np.array([0, 86, 51])
        # upper_red = np.array([179, 211, 239])
        # binary_mask = cv2.inRange(hsv, lower_red, upper_red)

        binary_mask_1 = cv2.inRange(hsv, lower_red1, upper_red1)
        binary_mask_2 = cv2.inRange(hsv, lower_red2, upper_red2)
        binary_mask = cv2.bitwise_or(binary_mask_1, binary_mask_2)
        


        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        # print(Image(mask).blobs)
        print(np.sum(mask)>100000)

        # im1 = cv2.bitwise_and(im, im, mask=mask)
        # mask = cv2.bitwise_or(binary_mask_1, binary_mask_2)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # im = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY)[1]

        # params = cv2.SimpleBlobDetector_Params()
        # # Change thresholds
        # params.minThreshold = 5
        # params.maxThreshold = 200

        # # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 15

        # # Filter by Circularity
        # params.filterByCircularity = False

        # # Filter by Convexity
        # params.filterByConvexity = False

        # # Filter by Inertia
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.01

        # # Create a detector with the parameters
        # detector = cv2.SimpleBlobDetector_create(params)

        # # Detect blobs
        # keypoints = detector.detect(im)

        # # Draw detected blobs as red circles
        # blank = np.zeros((1, 1))
        # blobs = cv2.drawKeypoints(im, keypoints, blank, (0, 0, 255),
        #                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # for idx, kp in enumerate(keypoints):
        #     print(f"Blob {idx+1}: Centroid = {kp.pt}, Area = {kp.size}")

        ##############################
        
        # im=im[:,:,60:240,:]

        #TO DO: apply any necessary image transforms
        #print(im)
        cv2.imshow("BotCam", im_cv)
        cv2.imshow("Stop Sign Mask", mask)
        cv2.waitKey(10)

        #TO DO: pass image through network get a prediction
        output_tensor = model(imagePreprocessing(transform(im_cv).unsqueeze(0)))
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
