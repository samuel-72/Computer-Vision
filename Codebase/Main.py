import os
import math
import numpy as np
import cv2
import FaceRecognizer
reload(FaceRecognizer)



face_cascade = cv2.CascadeClassifier('.\\Cascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('.\\Cascades\\haarcascade_eye.xml')
#mouth_cascade = cv2.CascadeClassifier('.\\Cascades\\smiled_01.xml')
mouth_cascade = cv2.CascadeClassifier('.\\Cascades\\haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier('.\\Cascades\\Nariz.xml')

def selectCascades():
    global face_cascade, eye_cascade, mouth_cascade, nose_cascade



def processImage(pathToImageFile,dataSet,typeOfDataSet):
    global face_cascade, eye_cascade, mouth_cascade, nose_cascade
    
    #Get the filename from the path - This will be the key for the DS storing the image features
    filename = pathToImageFile.split("\\")[len(pathToImageFile.split("\\"))-1].strip()
    print "File Being Processed : ", filename
    
    # Selecting the image and other setup for cv2
    img = cv2.imread(pathToImageFile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Input Image',img)
    # Identify the face in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    """
    print "Face is : ",faces
    """
    # Capture the cascade data in detectMultiScale, then draw the rectangles
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # Identify the eye, mouth and nose in the image
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, minNeighbors=10, minSize=(20,28))
        mouth = mouth_cascade.detectMultiScale(roi_gray,1.1, minNeighbors=10, minSize=(20,28))
        nose = nose_cascade.detectMultiScale(roi_gray,1.1, minNeighbors=10, minSize=(20,28))
        """
        print "Eyes are : ", eyes
        """
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)        
        
        for (ex,ey,ew,eh) in mouth:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            
        for (ex,ey,ew,eh) in nose:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(50,50,50),2)  
        
        # Store the Face, Eye, Nose and Mouth in their coressponding objects
        face     = FaceRecognizer.FR_Face(faces)
        leftEye  = FaceRecognizer.FR_Eye(eyes[1])
        rightEye = FaceRecognizer.FR_Eye(eyes[0])
        nose     = FaceRecognizer.FR_Nose(nose)
        mouth    = FaceRecognizer.FR_Mouth(mouth)
        """
        # Printing Debug information
        print "Leye Upper Left:        ", leftEye.upperLeftPoint
        print "Leye Upper Right:        ", leftEye.upperRightPoint
        print "Leye Lower Left:        ", leftEye.lowerLeftPoint
        print "Leye Lower Right:        ", leftEye.lowerRightPoint                        

        print "Reye Upper Left:        ", rightEye.upperLeftPoint
        print "Reye Upper Right:        ", rightEye.upperRightPoint
        print "Reye Lower Left:        ", rightEye.lowerLeftPoint
        print "Reye Lower Right:        ", rightEye.lowerRightPoint                        
        """
        # Calculating features
        distLeftRightEyeCenter  = calculateDistance(leftEye.center, rightEye.center)
        distLefEyeNoseCenter    = calculateDistance(leftEye.center, nose.center)
        distRightEyeNoseCenter  = calculateDistance(rightEye.center, nose.center)
        distLefEyeMouthCenter   = calculateDistance(leftEye.center, mouth.center)
        distRightEyeMouthCenter = calculateDistance(rightEye.center, mouth.center)
        
        #Storing features in object of class ImageFeatureSet
        if (typeOfDataSet == "Training Data"):
            dataSet.addToTrainingDataSet(filename,leftEye.center,rightEye.center,nose.center,mouth.center,distLeftRightEyeCenter,distLefEyeNoseCenter,distRightEyeNoseCenter,distLefEyeMouthCenter,distRightEyeMouthCenter,face.width)
        elif (typeOfDataSet == "Testing Data"):
            dataSet.addToTestDataSet(filename,leftEye.center,rightEye.center,nose.center,mouth.center,distLeftRightEyeCenter,distLefEyeNoseCenter,distRightEyeNoseCenter,distLefEyeMouthCenter,distRightEyeMouthCenter,face.width)
                    
        cv2.imshow("Image with identification", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def calculateDistance(point1, point2, method="euclidean"):
    if method == "euclidean":
            return ( math.sqrt( pow( (point1[0] - point2[0]),2 ) + pow( (point1[1] - point2[1]),2 )) )
    
def main():
    
    selectCascades()
    
    trainingData = FaceRecognizer.ImageFeatureSet()
    testingData = FaceRecognizer.ImageFeatureSet()

    # Call the below method in a loop for every file in the training data set
    for files in os.walk('.\\TrainingImages\\'):
        for filename in files[2]:
            processImage('.\\TrainingImages\\'+str(filename),trainingData,"Training Data")

    # Call the below method for extracting the features of the test image
    for files in os.walk('.\\TestImage\\'):
        for filename in files[2]:
            processImage('.\\TestImage\\'+str(filename),testingData,"Testing Data")

    print "\n\n\nPrinting the training data: \n******************************\n "
    trainingData.printDataSet("Training Data")    

    print "\n\n\nPrinting the test data: \n******************************\n"
    testingData.printDataSet("Testing Data")

if __name__ == '__main__':
    main()
