import numpy 
import matplotlib.pyplot as plt 
import cv2
import os 
import numpy as np
import shutil

day1Filenames=['TrackShort_Kd=10', 'TrackShort_Kd=15', 'Track0_Kd=5_Ka=15','Track0_Kd=10_Ka=25', 'Track0_Kd=20_Ka=25', 'Track1_Kd=5_Ka=15', 'TrackMed_Kd=10', 'TrackLong']
day2Filenames=['TrackLong_Kd=10', 'TrackSegments']
folder = f'./data/train/{day2Filenames[1]}'


def visualise(folderName=f'./data/train/EvenDistribution'):
    id=0
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if filename[-9]== '-':
                steer=filename[-9:][:-4]
            else: 
                steer=filename[-8:][:-4]    
            steering = np.float32(steer)
            
            print(filename, os.path.join(folderName, filename))
            
            save=False
            if steering == 0:
                id+=1
                # print(id)
            else:
                save=True

            if (id%4) == 0:
                save=True

            if save==True:
                print()
                shutil.copy(img_path, os.path.join(folderName, filename))
            
            


                


            # img = cv2.imread(img_path)
            
            # Display the image using matplotlib
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # if filename[-9]== '-':
            #     plt.title(filename[-9:])
            # else: 
            #     plt.title(filename[-8:])
            # plt.pause(0.1)

    # plt.show()
        
visualise()