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
