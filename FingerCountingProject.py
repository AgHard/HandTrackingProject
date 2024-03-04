import cv2
import time
import os
import numpy as np
import HandTrackingModule as htm

pTime = 0
wCam , hCam = 3000 , 2000
cap = cv2.VideoCapture(0)
cap.set(3 , wCam)
cap.set(4 , hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
#create list of images
    #import image
overLayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')#give path of image
    #print(f'{folderPath}/{imPath}')
    #save images
    overLayList.append(image)
print(len(overLayList))

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4 , 8 , 12 , 16 , 20]

while True:
    success , img = cap.read()
    img = detector.findHands(img)
    #create list of landmark that we detected
    lmList = detector.findPosition(img , draw=False)
    #print(lmList)
    if len(lmList) !=0:
        fingers = []
        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range (1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
    #to overlay an image the image is a matrix so what we can do we can define our new image we want
    # to put in our old image based on this location
        totalFingers = fingers.count(1)# how many ones
        print(totalFingers)

        h , w , c = overLayList[0].shape
        img[0:h , 0:w] = overLayList[totalFingers-1]

        cv2.rectangle(img , (20 , 255) , (170 , 425) , (0,255,0) , cv2.FILLED)
        cv2.putText(img , str(totalFingers) , (45 , 375) , cv2.FONT_HERSHEY_PLAIN ,
                    10 , (255,0,0) , 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FBS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image" , img)
    cv2.waitKey(1)