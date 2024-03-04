import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()# this class only uses rgp obj
mpDraw = mp.solutions.drawing_utils #draw points and lines

# doing frame rate and fps
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    #send rgp image to hands obj
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        # for every single hand draw points and connections
        for handLms in results.multi_hand_landmarks:
            # get id number and landmark info (x,y)
            for id , lm in enumerate (handLms.landmark):
                #print(id , lm)
                h , w , c = img.shape
                # its integer because its decimal places
                cx , cy = int(lm.x * w) , int(lm.y * h)
                print(id , cx , cy)
                #if id == 0:
                cv2.circle(img , (cx , cy) , 25 , (255,0,255) , cv2.FILLED)

            mpDraw.draw_landmarks(img , handLms , mpHands.HAND_CONNECTIONS)#draw image not rgb

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img , str(int(fps)),(10,70) , cv2.FONT_HERSHEY_PLAIN ,
        3 , (255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)