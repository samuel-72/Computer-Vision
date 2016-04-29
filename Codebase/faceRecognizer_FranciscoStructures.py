import numpy as np
import cv2
import FR_FacialObject
reload(FR_FacialObject)



#####################################################################
#################     HELPER FUNCTIONS    ###########################
#####################################################################

# just use this function to make FR_eyes
# Reye = FR_eyes[0]
# Leye = FR_eyes[1]
# this can be used with other facial objects, eg temples

def makeDoubleFacialObject(eyes):
    FR_eyes = []
    i= 0
    
    for item in eyes:
        eye_list = []
        eye = eyes[i]
        eye_list.append(eye)
        finalEye = FR_FacialObject.FR_FacialObject(eye_list)
        FR_eyes.append(finalEye)
        i += 1
    
    return FR_eyes
    
    
#####################################################################
#####################################################################




# select your cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('smiled_01.xml')
nose_cascade = cv2.CascadeClassifier('Nariz.xml')

# selecting the image and other setup for cv2
img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Capture the cascade data in detectMultiScale, then draw the rectangles
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(30,30))
    print eyes
    mouth = mouth_cascade.detectMultiScale(roi_gray,minSize=(30,40))
    nose = nose_cascade.detectMultiScale(roi_gray,minSize=(60,60))

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)        
        
    for (ex,ey,ew,eh) in mouth:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        
    for (ex,ey,ew,eh) in nose:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(50,50,50),2)


# this creates the FR_FacialObject
# FR stands for "Face Recognition"
# notice that for CascadeClassifer with two objects, like eyes, 
#     these have to be initialized with a special function



# create all the FacialObjects with FR_prefix
FR_eyes = makeDoubleFacialObject(eyes)
FR_nose = FR_FacialObject.FR_FacialObject(nose)
FR_mouth = FR_FacialObject.FR_FacialObject(mouth)


# FacialObject sample use cases
print "UPPER LEFT CORNER OF ALL FACIAL OBJECTS"
print "Reye:        ", FR_eyes[0].upperLeftPoint
print "Leye:        ", FR_eyes[1].upperLeftPoint
print "nose:        ", FR_nose.upperLeftPoint
print "mouth:       ", FR_mouth.upperLeftPoint

print "CENTER OF ALL FACIAL OBJECTS"
print "Reye:        ", FR_eyes[0].center
print "Leye:        ", FR_eyes[1].center
print "nose:        ", FR_nose.center
print "mouth:       ", FR_mouth.center



    


#cv2.imshow('img',img)
#cv2.waitKey(0)
cv2.destroyAllWindows()