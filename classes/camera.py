
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pprint 

pp = pprint.PrettyPrinter(indent=2, width=100)

# pp.pprint(sys.modules)
# pp.pprint(sys.path)

class Camera():
    def __init__(self, width, height):
    
        self.width        = width
        self.height       = height
        self.objectPts    = [] 
        self.ImagePts     = [] 
        
        self.calib_RMS    = None
        # Calibration Matrix
        self.cameraMatrix = None
        # Distortion coefficients k1, k2, p1, p2, k3
        self.distCoeffs   = None
        # Rotation vectors
        self.rotationVecs = None
        # Translation Vectors
        self.tranformVecs = None
        

    def calibrate(self, calImages):
        objectPts = [ image.objectPts for image in calImages]
        imagePts  = [ image.imagePts  for image in calImages]
        
        self.calib_RMS,    \
        self.cameraMatrix, \
        self.distCoeffs,   \
        self.rotationVecs, \
        self.transformVecs  = cv2.calibrateCamera(objectPts, imagePts, (self.width, self.height), None, None)
        
        print(' RMS reprojection error   : ', self.calib_RMS)
        print('\n Camera Matrix          : ', self.cameraMatrix.shape, '\n', self.cameraMatrix)
        print('\n Distortion Coefficients: ', self.distCoeffs)
        print('\n Rotation Vectors       : ', type(self.rotationVecs), len(self.rotationVecs) ,'  * ', self.rotationVecs[0].shape)
        
        for j in range(0,len(self.rotationVecs)):
            print('   ', j, ': ', self.rotationVecs[j].T)
        print('\n Transformation Vectors : ', type(self.transformVecs), len(self.transformVecs) ,'  * ', self.transformVecs[0].shape)
        
        for j in range(0,len(self.transformVecs)):
            print('   ', j, ': ', self.transformVecs[j].T)        
        
        return 

        
    def undistortImage(self, inputImg):
        return cv2.undistort(inputImg, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)
        
        
    def getImageChessboardCorners(self,img,nx,ny) :

        return cv2.findChessboardCorners(img, (nx, ny) ,None)
     
        
    
    
