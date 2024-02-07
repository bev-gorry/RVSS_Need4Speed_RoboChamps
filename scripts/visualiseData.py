import numpy 
import matplotlib.pyplot as plt 
import cv2
import os 

folder = './data/train/track2'

for filename in os.listdir(folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        # Display the image using matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if filename[-9]== '-':
            plt.title(filename[-9:])
        else: 
            plt.title(filename[-8:])
        plt.pause(0.1)

plt.show()