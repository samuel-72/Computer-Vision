import os

reload(os)
class Files:
    def __init__(self,path):
        self.fileList = []
        for files in os.walk(path):
            for filename in files[2]:
                self.fileList.append(filename)
        print ("Length is : ",str(len(self.fileList)))
        self.index = -1
    
    def __iter__(self):
        return self
        
    def next(self):
        
        if self.index==len(self.fileList)-1:
            raise StopIteration
        self.index=self.index+1
        print self.index
        return self.fileList[self.index]
        