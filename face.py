#-------------------------------------------------------------------#
# Author: Angel A Moreta
# All the credit to Viola And Jones face detection framework
# haarcascade files linked in description
#-------------------------------------------------------------------#
import cv2 as cv

#read image
original_image = cv.imread('people.jpg')

#Viola-jones alghithom
gray_scale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# Load the classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# detected parts
detected_eyes = eye_cascade.detectMultiScale(gray_scale_image)
detected_faces = face_cascade.detectMultiScale(gray_scale_image)
for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        ( column, row ),
        ( column + width, row + height ),
        ( 0, 0, 255 ),
        2 )
    for (c, r, width, height) in detected_eyes:
        cv.rectangle(
            original_image,
            ( c, r ),
            ( c + width, r + height ),
            ( 255, 0, 0 ),
            2 )

cv.imshow('Image', original_image)
cv.waitKey(0)
cv.destroyAllWindows()
