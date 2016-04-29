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



class FR_FacialObject:

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
        print type(self.upperLeftPoint)
        
        
    def getMidPoint(self, a, b):
        x_num = float(a[0] + b[0])
        y_num = float(a[1] + b[1])
        denom = 2
        
        midPoint = (x_num / denom, y_num / denom)
        return midPoint
        
        
    
            
    