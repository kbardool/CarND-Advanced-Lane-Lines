import numpy as np
import cv2
import matplotlib.pyplot as plt
import pprint 

pp = pprint.PrettyPrinter(indent=2, width=100)

class CalibrationImage():
    def __init__(self, filename, debug = False):

        try: 
            self.img       = cv2.imread(filename)
        except Exception as e:
            print(' Read failed for filename: ', filename)
            print(e)
        else:
            self.filename  = filename 
            self.channels  = 1 if self.img.ndim == 2 else self.img.shape[-1]
                           
            self.width     = self.img.shape[1]
            self.height    = self.img.shape[0]
            self.ImgSize   = self.img.shape[1::-1]
            self.nx        = None
            self.ny        = None
            self.cornersFnd = []
             
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

    def getImage(self, grayscale = False):
        if grayscale:
            return   cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2GRAY)
        else:
            return   cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2RGB)


    def findChessboardCorners(self,  nx, ny, img = None, save = False, debug = False) :

        assert (nx is not None) and (ny is not None), ' nx, ny must be specified'

        if self.img is None:
            try:
                imgGray = cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(' Read failed for filename: ', self.filename)
                print(e)
        else:
            imgGray = cv2.cvtColor(self.img , cv2.COLOR_BGR2GRAY)
            
        # Find the chessboard corners        
        ret, corners = cv2.findChessboardCorners(imgGray, (nx, ny),  cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE \
        + cv2.CALIB_CB_FAST_CHECK)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # objectpoint  coordiantes are in  (x=0:nx, y=0:ny, z=0)  form
        if debug:
            print('     findChessboardCorners() : ', self.filename)
            print('     Corners found: ', 0 if corners is None else corners.shape[0], ' Expected:   nx * ny: ', nx,'*', ny,'=',nx*ny, 
             '    return code: ',ret)

        if  save:
            self.nx         = nx
            self.ny         = ny 
            self.cornersFnd = ret
            self.corners    = corners
            self.cornersFound = 0 if  corners is None else corners.shape[0]
            objPts         = np.zeros((self.ny*self.nx,3), np.float32)
            objPts[:,:2]   = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)   
            self.objectPts = objPts
            self.imagePts  = np.squeeze(corners)
        
        # if debug:
            # print('Corners Found:', self.cornersFound) 
            # print(corners)
            # print('\nObject Pts:')
            # print(self.objectPts)
            # print('\nsqueeze(corners):', self.imagePts.shape)
            # print(self.imagePts)

        return ret
            
        
    def displayChessboardCorners(self, undistort = False, cam = None, nx = None, ny = None, debug = False):

        assert (not undistort) or  (undistort and cam is not None),"Camera must be provided when undistort == True"
        assert (self.nx is not None) or (nx is not None), "self.nx or nx cannot both be None"
        assert (self.ny is not None) or (ny is not None), "self.ny or ny cannot both be None"

        imgOrig = self.getImage()
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
    
        if undistort:
            imgResult = np.copy(cam.undistortImage(imgOrig))
            ret, corners = cv2.findChessboardCorners(cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY),(nx, ny))

        else: 
            imgResult = np.copy(imgOrig)
            ret, corners = self.cornersFnd, self.corners
        
        
        if debug:
            print(' displayChessboardCorners()')
            print('   nx: {}  ny: {}  imgResult.shape: {} '.format(nx, ny, imgResult.shape)) 
            if ret: 
                print('   Image  : ', self.filename, 'corners: ', corners.shape, ' ret    : ',ret, 'nx/ny: ', nx, ny)
            else:
                print('   Image  : ', self.filename, 'corners:  ---------    ret    : ',ret, 'nx/ny: ', nx, ny)
            
        imgResult = cv2.drawChessboardCorners(imgResult, (nx,ny), corners, ret)  
        
        return imgOrig, imgResult

         
# Draw and display the corners using CV
# img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
# cv2.imshow('img',img)
# cv2.waitKey(2500)
