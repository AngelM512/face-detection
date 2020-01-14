#angel moreta

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

#load classifier
load_cap = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')


while True:
    #capture frame by frame
    rec, frame = cap.read()
    if rec == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #detect face classifier
        dtcd_faces = load_cap.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            )
        for(x, y, h, w) in dtcd_faces:
            # capture a pic of the last frame sec
            print(x, y, h, w)
            ROI_pic = frame[y:y+h, x:x+w]
            pic_item = "face.png"
            cv.imwrite(pic_item, ROI_pic)

            #draw a rect at the ROI
            ROI_rect = cv.rectangle(frame,
                                    (x,y),
                                    (x+w, y+h),
                                    (255,0,0),
                                    4 )
        # CLOSE program if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        cv.imshow('frame', frame)



cap.release()
cv.destroyAllWindows()
