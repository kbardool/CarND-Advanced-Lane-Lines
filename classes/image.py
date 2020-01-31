import numpy as np
import cv2
import matplotlib.pyplot as plt
import pprint 

pp = pprint.PrettyPrinter(indent=2, width=100)

class CalibrationImage():
    def __init__(self, filename, debug = False):

        try: 
            img            = cv2.imread(filename)
        except Exception as e:
            print(' Read failed for filename: ', filename)
            print(e)
        else:
            self.filename  = filename 
            self.channels  = 1 if img.ndim == 2 else img.shape[-1]
                           
            self.width     = img.shape[1]
            self.height    = img.shape[0]
                           
            self.ImgSize   = img.shape[1::-1]
            self.nx        = []
            self.ny        = []
            self.PtrnFnd   = []
            
            # 3D points from real world space
            self.objectPts    = None
            # 2D points in image space
            self.imagePts     = None
            # Rotation Vectors 
            self.rotationVecs = None
            # Translation Vectors
            self.tranformVecs = None
            
            if debug:
                print( '\n Image object created for ', filename, 'h x w : ', self.height, ' x ' , 
                        self.width , ' image size : ', self.ImgSize, ' channels: ', self.channels) 


    def findChessboardCorners(self,  nx, ny, img = None, save = False, debug = False) :

        assert (nx is not None) and (ny is not None), ' nx, ny must be specified'
        if img is None:
            try:
                imgGray = cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(' Read failed for filename: ', self.filename)
                print(e)
        else:
            imgGray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
            
        # Find the chessboard corners        
        ret, corners = cv2.findChessboardCorners(imgGray, (nx, ny) ,None)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # objectpoint  coordiantes are in  (x=0:nx, y=0:ny, z=0)  form
        if debug:
            print(' displayChessboardCorners for : ', self.filename)
            if ret: 
                print(' Image  : ', self.filename, '               corners: ', corners.shape, ' ret    : ',ret, 'nx/ny: ', nx, ny)
            else:
                print(' Image  : ', self.filename, '               corners:  ---------    ret    : ',ret, 'nx/ny: ', nx, ny)
            

        if ret and save:
            self.nx        = nx
            self.ny        = ny 
            self.PtrnFnd   = ret
            objPts         = np.zeros((self.ny*self.nx,3), np.float32)
            objPts[:,:2]   = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)   
            self.objectPts = objPts
            self.imagePts  = np.squeeze(corners)
          
        return ret
            
    def getImage(self, grayscale = False):
        if grayscale:
            return   cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2GRAY)
        else:
            return   cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2RGB)


        
    def displayChessboardCorners(self, undistort = False, cam = None, nx = None, ny = None, debug = False):

        assert (not undistort) or  (undistort and cam is not None),"Camera must be provided when undistort == True"

        imgOrig = self.getImage()
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
    
        if undistort:
            imgResult = np.copy(cam.undistortImage(imgOrig))
        else: 
            imgResult = np.copy(imgOrig)
        print(nx, ny, imgResult.shape)
        
        ret, corners = cv2.findChessboardCorners(cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY),(nx, ny))

        if debug:
            print(' displayChessboardCorners for : ', self.filename)
            if ret: 
                print(' Image  : ', self.filename, 'corners: ', corners.shape, ' ret    : ',ret, 'nx/ny: ', nx, ny)
            else:
                print(' Image  : ', self.filename, 'corners:  ---------    ret    : ',ret, 'nx/ny: ', nx, ny)
            
        imgResult = cv2.drawChessboardCorners(imgResult, (nx,ny), corners, ret)  
        
        return imgOrig, imgResult

         
# Draw and display the corners using CV
# img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
# cv2.imshow('img',img)
# cv2.waitKey(2500)

