#angel moreta
import pickle
#----------------------------#
import cv2 as cv
import numpy as np

def nothing(x):
    pass

def trackbar(w_n):
    cv.createTrackbar('R', w_n, 0, 255, nothing)
    cv.createTrackbar('G', w_n, 0, 255, nothing)
    cv.createTrackbar('B', w_n, 0, 255, nothing)


def main():

    #load classifiers
    load_frontal_face = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    load_prof_face = cv.CascadeClassifier('haarcascade_profileface.xml')

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trainner.yml')

    #get labels from trainer
    labels = {}
    with open("labels.pickle", "rb") as f:
        or_labels = pickle.load(f)
        labels = {v:k for k,v in or_labels.items()}

    cap = cv.VideoCapture(0)

    #give it a win name
    w_nm = 'Recording'
    cv.namedWindow(w_nm)

    while True:
        #capture frame by frame
        rec, frame = cap.read()
        if rec == True:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            #detect face classifier
            frontal_face = load_frontal_face.detectMultiScale(
                gray,
                scaleFactor=1.5,
                minNeighbors=5,
                )
            for(x, y, h, w) in frontal_face:
                # capture a pic of the last frame sec
                print(x, y, h, w)
                roi_gray = gray[y:y+h, x:x+w]
                roi_frame = frame[y:y+h, x:x+w]

                pic_item = "face.png"
                cv.imwrite(pic_item, roi_frame)

                id_, conf = recognizer.predict(roi_gray)
                if conf >= 45:
                    print(labels[id_])

                #draw a rect at the ROI
                ROI_rect = cv.rectangle(frame,
                                        (x,y),
                                        (x+w, y+h),
                                        (255,0,0),
                                        4 )

            # CLOSE program if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            cv.imshow(w_nm, frame)

    # release + destroy windows
    cap.release()
    cv.destroyAllWindows()

main()
