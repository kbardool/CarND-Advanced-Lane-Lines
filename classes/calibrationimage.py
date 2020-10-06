import numpy as np
import cv2
import matplotlib.pyplot as plt
import pprint 

pp = pprint.PrettyPrinter(indent=2, width=100)

class CalibrationImage():
    def __init__(self, filename, debug = False):

        self.filename     = filename 
        
        try: 
            # self.imgOrig      = self.getImage(grayscale = False)
            self.imgOrig      = cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(' Read failed for filename: ', filename)
            print(e)
        else:
            self.channels     = 1 if self.imgOrig.ndim == 2 else self.imgOrig.shape[-1]                      
            self.width        = self.imgOrig.shape[1]
            self.height       = self.imgOrig.shape[0]
            self.imgSize      = self.imgOrig.shape[1::-1]    ## width x height
            # self.imgGray      = self.getImage(grayscale = True)
            self.imgGray      = cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2GRAY)
            self.nx           = None
            self.ny           = None
            self.cornersFnd   = False
            self.objectPts    = None       # 3D points from real world space
            self.imagePts     = None       # 2D points in image space       
            
            # Rotation Vectors 
            # self.rotationVecs = None
            # Translation Vectors
            # self.tranformVecs = None
            
            if debug:
                print( '\n Image object created for ', filename, 'h x w : ', self.height, ' x ' , 
                        self.width , ' image size : ', self.imgSize, ' channels: ', self.channels) 

    def getImage(self, grayscale = False):
        if grayscale:
            return   cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2GRAY)
        else:
            return   cv2.cvtColor(cv2.imread(self.filename) , cv2.COLOR_BGR2RGB)


    def findChessboardCorners(self,  nx, ny, img = None, save = True, debug = False) :
        '''

        '''
        assert (nx is not None) and (ny is not None), ' nx, ny must be specified'

        if img is None:
            img = self.imgGray 
            
        # Find the chessboard corners        
        ret, corners = cv2.findChessboardCorners(img, (nx, ny),  
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE \
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
            self.objectPts  = np.zeros((self.ny*self.nx,3), np.float32)
            self.objectPts[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)   
            self.imagePts   = np.squeeze(corners)
            self.corners    = corners
            # self.cornersFound = 0 if  corners is None else corners.shape[0]
        
        if debug:
            print('     Object Pts(3D)  :', self.objectPts.shape )
            print(self.objectPts)
            print('     Corners Detected:', self.cornersFnd) 
            print('self.imagePts: ', type(self.imagePts), ' \n', self.imagePts)
            print('     image Pts(2D)   :', type(self.imagePts), self.imagePts.shape if self.imagePts is not None else 'N/A')
            # print('     Corners Detected:', self.cornersFound) 
            # print(self.imagePts)
            # print('     corners         :', type(self.corners), self.corners.shape if self.corners is not None else 'N/A')
            # print(corners)
        return ret
            
        
    def displayChessboardCorners(self, undistort = False, cam = None, nx = None, ny = None, debug = False):

        assert (not undistort) or  (undistort and cam is not None),"Camera must be provided when undistort == True"
        assert (self.nx is not None) or (nx is not None), "self.nx or nx cannot both be None"
        assert (self.ny is not None) or (ny is not None), "self.ny or ny cannot both be None"

        imgOrig = self.imgOrig
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
    
        if undistort:
            # imgResult = np.copy(cam.undistortImage(imgOrig))
            imgResult = np.copy(cam.undistortImage(imgOrig))

            # ret, corners = cv2.findChessboardCorners(cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY),(nx, ny))
            # ret, corners = self.findChessboardCorners(nx = nx, ny=ny)
            ret, corners = cv2.findChessboardCorners(imgResult,(nx, ny))

        else: 
            imgResult = np.copy(self.imgOrig)
            ret, corners = self.cornersFnd, self.corners
        
        
        if debug:
            print(' displayChessboardCorners()')
            print('   nx: {}  ny: {}  imgResult.shape: {} '.format(nx, ny, imgResult.shape)) 
            if ret: 
                print('   Image  : ', self.filename, 'corners: ', corners.shape, ' ret    : ',ret, 'nx/ny: ', nx, ny)
            else:
                print('   Image  : ', self.filename, 'corners:  ---------    ret    : ',ret, 'nx/ny: ', nx, ny)

        ## Draw detected chessboard corners on result image    
        imgResult = cv2.drawChessboardCorners(imgResult, (nx,ny), corners, ret)  
        
        # return ret, imgOrig, imgResult
        return ret, imgResult
