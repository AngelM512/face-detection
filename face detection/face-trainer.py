


import os
import pickle
#-------------------------------------#
import cv2 as cv
from PIL import Image
import numpy as np

# load haarcascade and create a recognizer
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

# Do file navigation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR,"images")

# set up variables and empty dict + lists
SIZE = (512,512)
current_id = 0
label_ids = {}

x_train = []
y_labels = []

# loop for the root, desk and files in directory
for root, desk, files in os.walk(IMAGE_DIR):

    #loop throo desire file
    for file in files:

        # if photo ends with jpg,png or jpeg get the path and set a label.
        if file.endswith('jpg') or file.endswith('png') or\
        file.endswith('jpeg'):
            path = os.path.join(root, file)
            label = os.path.basename( root ).replace(" ","-").lower()

            if label in label_ids:
                pass # which is false
            else:
                #give each label an ID
                label_ids[label] = current_id
                current_id += 1

            # get ID number of the label
            id_ = label_ids[label]

            #convert image to grayscale
            pil_image = Image.open(path).convert("L") #grayscale
            pil_image = pil_image.resize(SIZE, Image.ANTIALIAS)
            image_array = np.array(pil_image, np.uint8)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,
            minNeighbors=5)

            for (x,y,w,h) in faces:
                #get the region of interest using the CascadeClassifier
                # with the given faces
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
