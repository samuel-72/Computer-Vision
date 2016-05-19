'''
This class makes it easy to reference the corners of the rectangles that
represent facial objects. 

FR stands for "Face Recognition"

Below is an example for a nose"

nose_cascade = cv2.CascadeClassifier('Nariz.xml')
nose = nose_cascade.detectMultiScale(roi_gray,minSize=(60,60))
f_nose = FR_FacialObject.FR_FacialObject(nose)
print f_nose.upperLeftPoint
print f_nose.lowerLeftPoint
print f_nose.upperRightPoint
print f_nose.lowerRightPoint

'''



class FR_Face:

    def __init__(self, facial_object):
    # these are all tuples
       
        self.upperLeftPoint         = (facial_object[0][0], facial_object[0][1])
        self.upperRightPoint        = (facial_object[0][2], facial_object[0][1])
        self.lowerLeftPoint	    = (facial_object[0][0], facial_object[0][3])
        self.lowerRightPoint	    = (facial_object[0][2], facial_object[0][3])
        
        self.leftMidPoint     = self.getMidPoint(self.upperLeftPoint, self.lowerLeftPoint)
        self.rightMidPoint    = self.getMidPoint(self.upperRightPoint, self.lowerRightPoint)
        self.topMidPoint      = self.getMidPoint(self.upperLeftPoint, self.upperRightPoint)
        self.bottomMidPoint   = self.getMidPoint(self.lowerLeftPoint, self.lowerRightPoint)
        self.center           = self.getMidPoint(self.upperLeftPoint, self.lowerRightPoint)
        self.width            = abs(float(self.upperLeftPoint[0]) - float(self.upperRightPoint[1]))
        
        
    def getMidPoint(self, a, b):
        x_num = float(a[0] + b[0])
        y_num = float(a[1] + b[1])
        denom = 2
        
        midPoint = (x_num / denom, y_num / denom)
        return midPoint
        
        
class FR_Eye(FR_Face):
    
        def __init__(self,eye_object):
        # these are all tuples
            #print "PE Cons", type(eye_object),  eye_object
            self.upperLeftPoint         = (eye_object[0], eye_object[1])
            self.upperRightPoint        = (eye_object[2], eye_object[1])
            self.lowerLeftPoint	        = (eye_object[0], eye_object[3])
            self.lowerRightPoint	= (eye_object[2], eye_object[3])
            #print "points : ", type(self.upperLeftPoint), self.upperLeftPoint
            
            self.leftMidPoint     = self.getMidPoint(self.upperLeftPoint, self.lowerLeftPoint)
            self.rightMidPoint    = self.getMidPoint(self.upperRightPoint, self.lowerRightPoint)
            self.topMidPoint      = self.getMidPoint(self.upperLeftPoint, self.upperRightPoint)
            self.bottomMidPoint   = self.getMidPoint(self.lowerLeftPoint, self.lowerRightPoint)
            self.center           = self.getMidPoint(self.upperLeftPoint, self.lowerRightPoint)
            #print type(self.upperLeftPoint)  , self.upperLeftPoint
            
        def getMidPoint(self, a, b):
            x_num = float(a[0]) + float(b[0])
            y_num = float(a[1]) + float(b[1])
            #print "x_num : ", x_num, " y_num :", y_num, "a,b : ", a, b
            denom = 2
        
            midPoint = (x_num / denom, y_num / denom)
            return midPoint
            

class FR_Nose(FR_Face):
    
        def __init__(self,nose_object):
            FR_Face.__init__(self,nose_object)    
        
class FR_Mouth(FR_Face):
    
        def __init__(self,mouth_object):
            FR_Face.__init__(self,mouth_object)    
        
class ImageFeatureSet:
    
    featureSet = ["Left Eye Center",
                            "Right Eye Center",
                            "Nose Center",
                            "Mouth Center",
                            "Distance between Left & Right Eye Centers",
                            "Distance between Left Eye & Nose Centers",
                            "Distance between Right Eye & Nose Centers",
                            "Distance between Left Eye & Mouth Centers",
                            "Distance between Right Eye & Mouth Centers",
                            "Face Width"
                           ]
        
    def __init__(self):
        self.trainingDataSet = {}
        self.testDataSet = {}
        
    def addToTrainingDataSet(self,filename,leftEyeCenter,rightEyeCenter,noseCenter,mouthCenter,distLeftRightEyeCenter,distLefEyeNoseCenter,distRightEyeNoseCenter,distLefEyeMouthCenter,distRightEyeMouthCenter,faceWidth):
        self.leftEyeCenter		=	leftEyeCenter
        self.rightEyeCenter		=	rightEyeCenter
        self.noseCenter	                =	noseCenter
        self.mouthCenter		=	mouthCenter
        self.distLeftRightEyeCenter	=	distLeftRightEyeCenter
        self.distLefEyeNoseCenter	=	distLefEyeNoseCenter
        self.distRightEyeNoseCenter	=	distRightEyeNoseCenter
        self.distLefEyeMouthCenter	=	distLefEyeMouthCenter
        self.distRightEyeMouthCenter	=	distRightEyeMouthCenter
        self.faceWidth			=	faceWidth
        self.trainingDataSet[filename]  =       (self.leftEyeCenter, self.rightEyeCenter, self.noseCenter, self.mouthCenter, self.distLeftRightEyeCenter, self.distLefEyeNoseCenter, self.distRightEyeNoseCenter, self.distLefEyeMouthCenter, self.distRightEyeMouthCenter, self.faceWidth)
        
    def addToTestDataSet(self,filename,leftEyeCenter,rightEyeCenter,noseCenter,mouthCenter,distLeftRightEyeCenter,distLefEyeNoseCenter,distRightEyeNoseCenter,distLefEyeMouthCenter,distRightEyeMouthCenter,faceWidth):
        self.leftEyeCenter		=	leftEyeCenter
        self.rightEyeCenter		=	rightEyeCenter
        self.noseCenter	                =	noseCenter
        self.mouthCenter		=	mouthCenter
        self.distLeftRightEyeCenter	=	distLeftRightEyeCenter
        self.distLefEyeNoseCenter	=	distLefEyeNoseCenter
        self.distRightEyeNoseCenter	=	distRightEyeNoseCenter
        self.distLefEyeMouthCenter	=	distLefEyeMouthCenter
        self.distRightEyeMouthCenter	=	distRightEyeMouthCenter
        self.faceWidth			=	faceWidth
        self.testDataSet[filename]      =       (self.leftEyeCenter, self.rightEyeCenter, self.noseCenter, self.mouthCenter, self.distLeftRightEyeCenter, self.distLefEyeNoseCenter, self.distRightEyeNoseCenter, self.distLefEyeMouthCenter, self.distRightEyeMouthCenter, self.faceWidth)
                        
    def printDataSet(self,dataSetToBePrinted):
        
        if dataSetToBePrinted == "Training Data":
            #print self.trainingDataSet.keys()
            #print self.trainingDataSet.values()
            for k in self.trainingDataSet.keys():
                print "\nFilename : ", k
                i = 0
                for item in self.trainingDataSet[k]:
                    print "{0:42s}  :  {1}".format(ImageFeatureSet.featureSet[i],item)
                    i = i + 1
                    
        elif dataSetToBePrinted == "Testing Data":
            for k in self.testDataSet.keys():
                print "\nFilename : ", k
                i = 0
                for item in self.testDataSet[k]:
                    print "{0:42s}  :  {1}".format(ImageFeatureSet.featureSet[i],item)
                    i = i + 1            
    
    
    def scoring(self,train,test,delta):
        print "training",train
        print "test for score",test
        #Sample testing data delete later
        # test = {"testing": ((48.0, 60.0), (93.0, 59.0), (84.0, 88.5), (82.0, 105.0), 45.0111097397076, 45.9156835950419, 30.84234102658227, 56.4003546088143, 47.29693436154187, 49.0)}
        #The dictionary will store the confidence score for a particular test image when trained by a particular image
        confidence_score ={}
        
        #Iterating through each test data and calculating the confidence score for each data in the training set
        for key_test,value_test in test.iteritems():  
            
            for key_train,value_train in train.iteritems():
                confi={}
                i=0;
                #confident increments when there is a matching attribute 
                confident=0;
                #The number of matching features gives a count of the number of features which are matched
                number_of_matching=0;
                
                for y in value_train:
                    if(i<4):
                        # Matching each feature and calculating the confidence score 
                        if(abs((y[0]-value_test[i][0]))<delta and abs((y[1]==value_test[i][1])))<delta :
                            confident = confident+10
                            number_of_matching=number_of_matching+1
                        i=i+1
                    else:
                        if(abs((y-value_test[i]))<delta):
                            confident = confident+10
                            number_of_matching=number_of_matching+1
                            
                        i=i+1
                            
                                                
                            
                            
                confi[key_train] =[confident,number_of_matching]
                confidence_score[key_test]=confi
                
                print confi
                
                #Printing out the output to a text file output.txt            
                with open ("output.txt","a") as fh:
                    for x,y in confidence_score.iteritems():
                        for h,w in y.iteritems():
                            out="Testing Image-->   %s   Training Image-->   %s   Confidence Score-->    %d    Number of Matching attributes-->   %d\n" % (x,h,w[0],w[1])
                            fh.write(out)
                       
        return confidence_score
 
    
                        
                         
