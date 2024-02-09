# TO run this script, enter a frame path as cmd line arg: python3 color_thresholder_app.py frame01.png
import cv2
import numpy as np
import sys
import os
import argparse
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot
import torchvision.transforms as transforms
from machinevisiontoolbox import Image

def nothing(x):
  pass

cap = cv2.VideoCapture(0)
#Set the camera resolution
cap.set(3, 320)                     	# Set the width to 320
cap.set(4, 240)                     	# Set the height to 240


cv2.namedWindow('Thresholder_App')


cv2.createTrackbar("VMax", "Thresholder_App",0,255,nothing)
cv2.createTrackbar("VMin", "Thresholder_App",0,255,nothing)
cv2.createTrackbar("SMax", "Thresholder_App",0,255,nothing)
cv2.createTrackbar("SMin", "Thresholder_App",0,255,nothing)
cv2.createTrackbar("HMax", "Thresholder_App",0,179,nothing)
cv2.createTrackbar("HMin", "Thresholder_App",0,179,nothing)

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
args = parser.parse_args()

transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

bot = PiBot(ip=args.ip)
bot.setVelocity(0, 0)


cv2.setTrackbarPos("VMax", "Thresholder_App", 255)
cv2.setTrackbarPos("VMin", "Thresholder_App", 0)
cv2.setTrackbarPos("SMax", "Thresholder_App", 255)
cv2.setTrackbarPos("SMin", "Thresholder_App", 0)
cv2.setTrackbarPos("HMax", "Thresholder_App", 179)
cv2.setTrackbarPos("HMin", "Thresholder_App", 0)


while(1):
  
   vmax=cv2.getTrackbarPos("VMax", "Thresholder_App")
   vmin=cv2.getTrackbarPos("VMin", "Thresholder_App")
   smax=cv2.getTrackbarPos("SMax", "Thresholder_App")
   smin=cv2.getTrackbarPos("SMin", "Thresholder_App")
   hmax=cv2.getTrackbarPos("HMax", "Thresholder_App")
   hmin=cv2.getTrackbarPos("HMin", "Thresholder_App")
   

   min_ = np.array([hmin,smin,vmin])
   max_ = np.array([hmax,smax,vmax])

   #ret, frame = cap.read()
   #cv2.imshow('frame',frame)
   img = bot.getImage()
  #  img=transform(img).unsqueeze(0)
   img=img[60:240,:]
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   gray=cv2.bitwise_not(cv2.threshold(gray,128, 255,cv2.THRESH_BINARY)[1])
  #  contours=cv2.threshold(gray)
  #  blobs = gray.blobs()
   
  #  blobs=Image(gray).blobs()
  #  print(blobs)

   
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

   mask = cv2.inRange(hsv, min_, max_)

   thresholded_img = cv2.bitwise_and(img, img, mask= mask)

   
   cv2.imshow("Thresholder_App",thresholded_img)

   k = cv2.waitKey(1) & 0xFF

   # exit if q or esc are pressed
   if (k == ord('q') or k == 27):
     break
   elif (k == ord('p')):
     print("Min HSV ["+ str(min_[0]) + ","+str(min_[1])+","+str(min_[2])+"]")    
     print("Max HSV ["+ str(max_[0]) + ","+str(max_[1])+","+str(max_[2])+"]")
     print("[np.array(["+ str(min_[0]) + ","+str(min_[1])+","+str(min_[2])+"]),np.array(["+ str(max_[0]) + ","+str(max_[1])+","+str(max_[2])+"])]") 

cv2.destroyAllWindows()
