import numpy as np
import cv2
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from classes.plotting import PlotDisplay
from .utils import get_ax, display_one, display_two, display_multi
deg2rad = lambda  x: x * np.pi/180
rad2deg = lambda  x: x * 180 / np.pi




def convert_to_gray(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.copy(img)
    return gray

def perspectiveTransform(img, source, dest, debug = False):

    M = cv2.getPerspectiveTransform(source, dest)
    Minv = cv2.getPerspectiveTransform(dest, source)    

    if debug: 
        print('src: ', type(source), source.shape, ' - ',  source)
        print('dst: ', type(dest), dest.shape, ' - ',  dest)
        print(' M: ', M.shape, ' Minv: ', Minv)
        
    return cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR), M, Minv


def unwarpImage(img, nx, ny, cam, dst, debug = False):
    # 1) Undistort using mtx and dist
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found: 
            # a) draw corners
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    
    ## 1. UNDISTORT 
    undist = cv2.undistort(img, cam.cameraMatrix, cam.distCoeffs, None, cam.cameraMatrix)
    
    ## 2. Convert to Gray Scale
    imgGray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    ## 3. Find corners
    ret, corners = cv2.findChessboardCorners(imgGray, (nx, ny), cv2.CALIB_CB_ADAPTIVE_THRESH)
    if ret:
        if debug:
            print(' img shape: ',undist.shape[1::-1])
            print(' ret    : ', ret)
            if ret:
                print(' Corners: ', corners.shape)
                # print(corners)

        cornerCount = corners.shape[0]
        if (cornerCount % nx) != 0:
            print(' Not all corners have been detected!! nx:', nx, ' ny: ', ny, ' corners detected: ', cornerCount)
            M = np.eye(3)
        else:
            ## 4a. if corners found Draw corners 
            cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)        
            
            ## 4b. define 4 source pointsl        
            src = np.float32([corners[0,0], corners[nx-1,0], corners[nx*(ny-1),0], corners[nx*ny-1,0]])

            ## 4c. define 4 destination points
            
            ## 4d. get M tranform matrix     
            M = cv2.getPerspectiveTransform(src, dst)

            if debug: 
                print('src: ', type(src), src.shape, ' - ',  src)
                print('dst: ', type(dst), dst.shape, ' - ',  dst)
                print(' M:', M.shape)
    else:
        M = np.eye(3)
        
    ## 4e. warp image to top-down view
    warped = cv2.warpPerspective(undist, M, undist.shape[1::-1], flags=cv2.INTER_LINEAR)
    return undist, warped, M

    
def erodeImage(img, ksize = 7, iters = 1):
    # Eroding the image , decreases brightness of image
    # Get structuring element/kernel which will be used for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))
    result = cv2.erode(img, kernel, iterations = iters)
    
    return  result

def dilateImage(img, ksize = 7, iters = 1):
    # Eroding the image , decreases brightness of image
    # Get structuring element/kernel which will be used for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))
    result = cv2.dilate(img, kernel, iterations = iters)
    return  result


def erodeDilateImage(img, ksize = 7, iters = 1):
    # Eroding the image , decreases brightness of image
    # Get structuring element/kernel which will be used for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))
    result = cv2.erode(img, kernel, iterations = iters)
    result = cv2.dilate(result, kernel, iterations = iters)
    return  result

def openImage(img, ksize = 3, iters = 1):
    openingSize = ksize
    # Selecting a elliptical kernel
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * openingSize + 1, 2 * openingSize + 1), (openingSize,openingSize))
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, element, iterations = iters)
    return result

def closeImage(img, ksize = 3, iters = 1):
    openingSize = ksize
    # Selecting a elliptical kernel
    # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * openingSize + 1, 2 * openingSize + 1), (openingSize,openingSize))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize), (-1 , -1))
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element, iterations = iters)
    return result

def adjust_contrast(image, alpha=1.0, beta = 0, debug = False):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    
    result = np.copy(image)
    alpha_mat = np.ones_like(image)*alpha
    result = np.clip(alpha_mat*result + beta, 0, 255).astype(np.uint8)
    if debug:
        print(' image: ', image.max(), image.min(), image.dtype)
        print(' image: ', result.max(), result.min(), result.dtype)
        display_two(image  , result    , title1 = 'imgUndist', title2 = ' imgAdjust')
    return result

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def grad_abs_thresh(img_gray, orient='x', sobel_kernel=3 , thresh=(0, 255), display = False):
    '''
    2) Take the derivative in x or y given orient = 'x' or 'y'
    3) Take the absolute value of the derivative or gradient
    4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    5) Create a mask of 1's where the scaled gradient magnitude 
       is > thresh_min and < thresh_max
    6) Return this mask as your binary_output image
    '''
    # gray = convert_to_gray(img)
    
    x_sobel = 1 if orient == 'x' else 0
    y_sobel = 1 if x_sobel == 0 else 0
    sobel = cv2.Sobel(img_gray, cv2.CV_64F, x_sobel, y_sobel, ksize = sobel_kernel)
        
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / abs_sobel.max())
    
    # binary_output = np.zeros_like(scaled_sobel)
    # binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    binary_output = ((scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])).astype(np.uint8)
    
    thresh_str = str(thresh)
        
    if display:
        # print('sobel       : ', sobel.shape, sobel.min(), sobel.max())
        # print('scaled_sobel: ', scaled_sobel.shape, scaled_sobel.min(), scaled_sobel.max())
        # print('abs_sobel   : ', abs_sobel.shape, abs_sobel.min(), abs_sobel.max())    
        print('grad_abs_thresh(): input:', img_gray.shape, ' min: ', img_gray.min(), ' max: ', img_gray.max())
        filtered_sobel = np.copy(scaled_sobel)
        filtered_sobel[(scaled_sobel < thresh[0]) | (scaled_sobel > thresh[1])] = 0
        display_one(scaled_sobel  , title='scaled_sobel - orientation: '+orient+'  thresholds: '+thresh_str, cmap='jet')
        display_one(filtered_sobel, title='filtered_sobel - orientation: '+orient+' within thresholds: '+thresh_str, cmap='jet')
        display_one(binary_output , title='result image - orientation: '+orient+'  thresholds: '+thresh_str, cbar =True)
    return binary_output

def grad_x_thresh(img_gray, sobel_kernel=3 , thresh=(0, 255), display = False):
    '''
    Theresholding on derivative in X orientation 
    '''
    abs_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / abs_sobel.max())
    
    binary_output = ((scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])).astype(np.uint8)
    
    thresh_str = str(thresh)
    
    if display:
        # print('sobel       : ', sobel.shape, sobel.min(), sobel.max())
        # print('scaled_sobel: ', scaled_sobel.shape, scaled_sobel.min(), scaled_sobel.max())
        # print('abs_sobel   : ', abs_sobel.shape, abs_sobel.min(), abs_sobel.max())    
        print('grad_abs_thresh(): input:', img_gray.shape, ' min: ', img_gray.min(), ' max: ', img_gray.max())
        filtered_sobel = np.copy(scaled_sobel)
        filtered_sobel[(scaled_sobel < thresh[0]) | (scaled_sobel > thresh[1])] = 0
        display_one(scaled_sobel  , title='scaled_sobel  - thresholds: '+thresh_str, cmap='jet', cbar =True)
        display_one(filtered_sobel, title='filtered_sobel- within    : '+thresh_str, cmap='jet', cbar =True)
        display_one(binary_output , title='result image  - thresholds: '+thresh_str, cbar =True)
    return binary_output

def grad_y_thresh(img_gray, sobel_kernel=3 , thresh=(0, 255), display = False):
    '''
    Theresholding on derivative in Y orientation 
    '''
    abs_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / abs_sobel.max())
    
    binary_output = ((scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])).astype(np.uint8)
   
    return binary_output


def grad_mag_thresh(img_gray, sobel_kernel=3, thresh=(0, 255), display = False):
    '''
    Define a function that applies Sobel x and y, 
    then computes the magnitude of the gradient
    and applies a threshold
    
    Apply the following steps to img
     2) Take the gradient in x and y separately
     3) Calculate the magnitude 
     4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
     5) Create a binary mask where mag thresholds are met
     6) Return this mask as your binary_output image
    ''' 
    
    thresh_min, thresh_max = thresh
    
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)    
    
    # 3) Calculate the magnitude 
    abs_sobel = np.sqrt((np.absolute(sobel_x )** 2) + (np.absolute(sobel_y) ** 2))
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / abs_sobel.max())
    
    # 5) Create a binary mask where mag thresholds are met
    # binary_output = np.zeros_like(scaled_sobel)
    # binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    binary_output = ((scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])).astype(np.uint8)
    
    thresh_str = str(thresh)
    
    if display:
        filtered_sobel = np.copy(scaled_sobel)
        filtered_sobel[(scaled_sobel < thresh[0]) | (scaled_sobel > thresh[1])] = 0
        display_one(scaled_sobel  , title='scaled_sobel - thresholds: '+thresh_str, cmap='jet', cbar =True)
        display_one(filtered_sobel, title='filtered_sobel - within thresholds: '+thresh_str, cmap='jet', cbar =True)
        display_one(binary_output , title='result image - thresholds: '+thresh_str, cbar =True)    
    
    return binary_output

def grad_dir_thresh(img_gray, sobel_kernel=3, thresh=(0,90), display = False):
    '''
    applies Sobel x and y, then computes the direction of the gradient
    and applies a threshold.

    '''
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image

    thresh_rad_low  = deg2rad(thresh[0]) 
    thresh_rad_high = deg2rad(thresh[1])
    
    abs_sobel_x = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    abs_sobel_y = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    
    grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
    grad_dir_deg = np.rad2deg(grad_dir)
    
    binary_output = ((grad_dir >= thresh_rad_low) & (grad_dir <= thresh_rad_high)).astype(np.uint8)
    
    if display:
        # print("grad_dir:", grad_dir.shape, ' Deg: ', thresh_str + ' Rad: ' + str(thresh_rads))
        thr_str = '( {:5.1f}, {:5.1f})'.format(thresh[0], thresh[1])
        thr_rads_str ='( {:5.3f}, {:5.3f})'.format(thresh_rad_low, thresh_rad_high)

        cmin = grad_dir.min(); cmax = grad_dir.max()
        cmin_deg = grad_dir_deg.min(); cmax_deg = grad_dir_deg.max()

        filtered_dir = np.copy(grad_dir)
        filtered_dir[(grad_dir < thresh_rad_low) | (grad_dir > thresh_rad_high)] = 0
        filtered_dir_deg = np.copy(grad_dir_deg)
        filtered_dir_deg[(grad_dir_deg < thresh[0]) | (grad_dir_deg > thresh[1])] = 0

        
        display_one(grad_dir    , title='gradient direction (Radians)' , cbar = True)
        display_one(grad_dir_deg, title='gradient direction (Degrees)' , cbar = True)
        display_one(filtered_dir, title='filtered_grad (rad) - w/i range Deg: '+thr_str+ '   Rad:'+thr_rads_str, 
                                      clim = (cmin, cmax), cbar = True)
        display_one(filtered_dir_deg, title='filtered_grad (deg) - w/i range Deg: '+thr_str+ '   Rad:'+thr_rads_str, 
                                      clim = (cmin_deg, cmax_deg), cbar = True)
        display_one(binary_output   , title='result image - thresholds: Deg: '+thr_str+ '   Rad:'+thr_rads_str)
                                      

    return binary_output

def color_thresh(img, thresh=(0, 255), channel = 0, display = False):
    s_channel = img[:,:,channel]
    
    binary_output = ((s_channel >= thresh[0]) & (s_channel <= thresh[1])).astype(np.uint8)

    thresh_str = str(thresh)
    
    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        print(' filtered_dir : ', filtered_dir.shape)
        print(' s_channel    : ', s_channel.shape)
        print(' binary_output: ', binary_output.shape)
        
        display_one(s_channel, title='RGB channel '+str(channel)+ '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()), cbar = True )    
        display_one(filtered_dir, title='filtered_grad - within thresholds: '+thresh_str)        
        display_one(binary_output, title='Thresholded Output- thresholds: '+thresh_str)    
    
    return binary_output

def RGB_thresh(img, thresh=(0, 255),  display = False):
    s_channel = np.copy(img)
    red_binary   = color_thresh(s_channel, thresh = thresh, channel = 0 ,display = display)
    green_binary = color_thresh(s_channel, thresh = thresh, channel = 1, display = display)
    blue_binary  = color_thresh(s_channel, thresh = thresh, channel = 2, display = display)
 
    binary_output = ((red_binary == 1) | (green_binary == 1)| (blue_binary ==1)).astype(np.uint8) 

    thresh_str = str(thresh)
    
    if display:
        
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        # filtered_dir = s_channel * np.dstack((binary_output,binary_output,binary_output))
   
        print(' filtered_dir : ', filtered_dir.shape)
        print(' s_channel    : ', s_channel.shape)
        print(' binary_output: ', binary_output.shape)
        display_one(s_channel, title='RGB channels   Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) )    
        display_one(filtered_dir , title='filtered_grad - within thresholds: '+thresh_str+' Min: ' +str(filtered_dir.min())+ ' Max: '+str(filtered_dir.max()) )        
        print(binary_output.shape, binary_output.min(), binary_output.max())
        display_one(binary_output, title='Thresholded Output - thresholds: '+thresh_str)    
    
    return binary_output
    

def hue_thresh(img_hls, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    # hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # binary_output = np.zeros_like(s_channel)
    # binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 
    
    s_channel = img_hls[:,:,0]    
    binary_output = ((s_channel >= thresh[0]) & (s_channel <= thresh[1])).astype(np.uint8)
    
    thresh_str = str(thresh)
    
    if display:
        # filtered_dir = np.zerosl(s_channel)
        filtered_dir = ((s_channel < thresh[0]) | (s_channel > thresh[1])).astype(np.uint8)
        
        display_one(s_channel    , title='Hue channel '+thresh_str + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()), cbar =True )    
        display_one(filtered_dir , title='filtered_grad - outside thresholds: '+thresh_str, cbar =True, cmap = 'gray')        
        display_one(binary_output, title='Thresholded Output- thresholds: '+thresh_str, cbar =True, cmap = 'gray')    
        
    return binary_output

def level_thresh(img_hls, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    # hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # binary_output = np.zeros_like(s_channel)
    # binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    thresh_str = str(thresh)

    s_channel = img_hls[:,:,1]
    binary_output = ((s_channel >= thresh[0]) & (s_channel <= thresh[1])).astype(np.uint8)
    
    
    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one(s_channel, title='Level channel '+thresh_str + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()), cbar =True )    
        display_one(filtered_dir, title='filtered_grad - within thresholds: '+thresh_str, cbar =True)        
        display_one(binary_output, title='Thresholded Output- thresholds: '+thresh_str, cbar =True)    
    
    # ttl_pxls  = (img_hls.shape[0] * img_hls.shape[1])
    # pxl_ratio = round(np.sum(binary_output) * 100/ttl_pxls,2)
    # binary_output = binary_output if pxl_ratio > 50 else np.zeros_like(binary_output)
    
    return binary_output

def saturation_thresh(img_hls, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    # hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # binary_output = np.zeros_like(s_channel)
    # binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    s_channel = img_hls[:,:,2]
    binary_output = ((s_channel >= thresh[0]) & (s_channel <= thresh[1])).astype(np.uint8)

    thresh_str = str(thresh)
    
    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one(s_channel, title='Saturation channel '+thresh_str + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()), cbar = True )    
        display_one(filtered_dir, title='filtered_grad - within thresholds: '+thresh_str, cbar = True)        
        display_one(binary_output, title='Thresholded Output- thresholds: '+thresh_str, cbar = True)    
    
    return binary_output



def HSV_value_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = hlsImage[:,:,-1]

    # binary_output = np.zeros_like(s_channel)
    # binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    
    binary_output = ((s_channel >= thresh[0]) & (s_channel <= thresh[1])).astype(np.uint8)

    thresh_str = str(thresh)
    
    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one(s_channel, title='Level channel '+thresh_str + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()), cbar =True )    
        display_one(filtered_dir, title='filtered_grad - within thresholds: '+thresh_str, cbar =True)        
        display_one(binary_output, title='Thresholded Output- thresholds: '+thresh_str, cbar =True)    
    return binary_output

def YCrCb_Y_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to  YCrCb color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    s_channel = hlsImage[:,:,0]

    # binary_output = np.zeros_like(s_channel)
    # binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    
    binary_output = ((s_channel >= thresh[0]) & (s_channel <= thresh[1])).astype(np.uint8)

    thresh_str = str(thresh)
    
    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one(s_channel, title='Level channel '+thresh_str + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()), cbar =True )    
        display_one(filtered_dir, title='filtered_grad - within thresholds: '+thresh_str, cbar =True)        
        display_one(binary_output, title='Thresholded Output- thresholds: '+thresh_str, cbar =True)    
    return binary_output

def YCrCb_Cr_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to  YCrCb color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    s_channel = hlsImage[:,:,1]

    # binary_output = np.zeros_like(s_channel)
    # binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    
    binary_output = ((s_channel >= thresh[0]) & (s_channel <= thresh[1])).astype(np.uint8) 

    thresh_str = str(thresh)
    
    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one(s_channel, title='Level channel '+thresh_str + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()), cbar =True)    
        display_one(filtered_dir, title='filtered_grad - within thresholds: '+thresh_str, cbar =True)        
        display_one(binary_output, title='Thresholded Output- thresholds: '+thresh_str, cbar =True)    
    return binary_output

def YCrCb_Cb_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to YCrCb color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    s_channel = hlsImage[:,:,-1]

    # binary_output = np.zeros_like(s_channel)
    # binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    
    binary_output = ((s_channel >= thresh[0]) & (s_channel <= thresh[1])).astype(np.uint8)

    thresh_str = str(thresh)
    
    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one(s_channel, title='Level channel '+thresh_str + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()), cbar =True )    
        display_one(filtered_dir, title='filtered_grad - within thresholds: '+thresh_str, cbar =True)        
        display_one(binary_output, title='Thresholded Output- thresholds: '+thresh_str, cbar =True)    
    return binary_output

def apply_thresholds(img,  thrshlds , **kwargs): 
 
    thrshlds.setdefault('ksize'  , 10)
    thrshlds.setdefault('x_thr'  , 0)
    thrshlds.setdefault('y_thr'  , None)
    thrshlds.setdefault('mag_thr', None)
    thrshlds.setdefault('dir_thr', None)
    thrshlds.setdefault('rgb_thr', None)
    thrshlds.setdefault('lvl_thr', None)
    thrshlds.setdefault('sat_thr', None)
    thrshlds.setdefault('hue_thr', None)
    
    ret    = kwargs.get('ret'   , None)
    debug  = kwargs.get('debug' , False)
    debug2 = kwargs.get('debug2', False)
    results    = {}
    ttl_pixels = img.shape[0] * img.shape[1]
    half_pxls  = ttl_pixels // 2    
    img_gray   = convert_to_gray(img)
    img_hls    = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    empty_shape = np.zeros((img.shape[0], img.shape[1]),dtype = np.uint8)

    # Apply each of the thresholding functions
    results['cmb_x']   = grad_x_thresh(img_gray,  sobel_kernel=thrshlds['ksize'], thresh=thrshlds['x_thr'])
    
    if thrshlds['y_thr']  is None:
        results['cmb_y'] = np.copy(empty_shape)
    else:
        results['cmb_y'] = grad_y_thresh(img_gray, sobel_kernel=thrshlds['ksize'], thresh=thrshlds['y_thr'])

    if thrshlds['mag_thr'] is None :
        mag_thr = np.copy(empty_shape)
    else:
        mag_thr = grad_mag_thresh(img_gray, sobel_kernel=thrshlds['ksize'], thresh=thrshlds['mag_thr'])
    
    if thrshlds['dir_thr'] is None:
        dir_thr = np.copy(empty_shape)
    else:
        dir_thr  = grad_dir_thresh(img_gray, sobel_kernel=thrshlds['ksize'], thresh=thrshlds['dir_thr'])

    results['cmb_mag'] = mag_thr & dir_thr 

    if thrshlds['rgb_thr'] is None:
        results['cmb_rgb']= np.copy(empty_shape)
    else:
        results['cmb_rgb'] = RGB_thresh(img, thresh = thrshlds['rgb_thr'])

    ## Thresholds based on the HLS color model        

    if thrshlds['lvl_thr'] is None:
        results['cmb_lvl'] = np.copy(empty_shape)
    else:
        results['cmb_lvl'] = level_thresh(img_hls, thresh = thrshlds['lvl_thr'])
        
    if thrshlds['sat_thr'] is None:
        results['cmb_sat'] = np.copy(empty_shape)
    else:
        results['cmb_sat'] = saturation_thresh(img_hls, thresh = thrshlds['sat_thr'])
    
    if thrshlds['hue_thr'] is None:
        results['cmb_hue'] = np.copy(empty_shape)
    else:
        results['cmb_hue'] = hue_thresh(img_hls, thresh = thrshlds['hue_thr'])

    pxl_ratio = {}; hlf_ratio = {}
    for k in results.keys():
        hlf_ratio[k] = round(np.sum(results[k][450:680,:]) * 100/half_pxls,2)
        pxl_ratio[k] = round(np.sum(results[k]) * 100/ttl_pixels,2)
        if hlf_ratio[k] > 9.0:
            results[k] = np.zeros_like(results[k]) 
            if debug:
                print(' set {} to zeros hlf_pixel: {}     pxl_ratio: {}'.format( k, hlf_ratio[k], pxl_ratio[k]))

    results['cmb_xy'] = results['cmb_x'] | results['cmb_y']
    results['cmb_mag_x'] = results['cmb_mag'] | results['cmb_x']  
    results['cmb_mag_xy'] = results['cmb_mag'] | results['cmb_xy']   
    results['cmb_mag_lvl_x'] = results['cmb_mag_x'] | results['cmb_lvl']  
    results['cmb_mag_sat_lvl_x'] = results['cmb_mag_lvl_x'] | results['cmb_sat']  
    # results['cmb_mag_xy'] = results['cmb_mag_x'] | results['cmb_y']  

    results['cmb_rgb_lvl'] = results['cmb_rgb']  | results['cmb_lvl']  
    results['cmb_rgb_lvl_sat'] = results['cmb_rgb_lvl'] | results['cmb_sat']  
    results['cmb_rgb_mag_x'] = results['cmb_rgb']   | results['cmb_mag'] | results['cmb_x']
    results['cmb_rgb_lvl_sat_mag'] = results['cmb_rgb_lvl_sat'] |  results['cmb_mag']  
    results['cmb_rgb_lvl_sat_mag_x'] = results['cmb_rgb_lvl_sat_mag'] | results['cmb_x']
    # results['cmb_rgb_lvl_sat_mag_xy'] = results['cmb_rgb_lvl_sat_mag_x'] | results['cmb_y'] 

    results['cmb_hue_x'] = results['cmb_hue']  | results['cmb_x']  
    results['cmb_hue_mag_x'] = results['cmb_hue_x'] | results['cmb_mag']  
    results['cmb_hue_lvl_x'] = results['cmb_hue_x'] | results['cmb_lvl']  
    results['cmb_hue_mag_lvl_x'] = results['cmb_hue_mag_x'] | results['cmb_lvl']  
    results['cmb_hue_mag_sat'] = results['cmb_hue'] | results['cmb_mag'] | results['cmb_sat']
    results['cmb_hue_mag_lvl_sat'] = results['cmb_hue'] | results['cmb_mag'] |  results['cmb_lvl'] | results['cmb_sat']

    results['cmb_sat_x'] = results['cmb_sat']  | results['cmb_x']  
    results['cmb_sat_mag'] = results['cmb_sat']  | results['cmb_mag']  
    results['cmb_sat_mag_x'] = results['cmb_sat_mag']  | results['cmb_x']  

    if debug2:
        # fig = plt.figure(figsize=(25,28))
        fig, axs = plt.subplots(5,2,figsize=(25,28))
        axs[0,0].imshow(img)                             ;  axs[0,0].set_title(' input image: '+str(img.shape[0])+' x '+str(img.shape[1]))
        axs[0,1].imshow(results['cmb_x']  , cmap ='gray');  axs[0,1].set_title('grad x thresholding: '     + str(thrshlds['x_thr']))
        axs[1,0].imshow(results['cmb_y']  , cmap ='gray');  axs[1,0].set_title('grad y thresholding: '     + str(thrshlds['y_thr'])) 
        axs[1,1].imshow(results['cmb_sat'], cmap ='gray');  axs[1,1].set_title('Saturation thresholding '  + str(thrshlds['sat_thr']))
        axs[2,0].imshow(results['cmb_lvl'], cmap ='gray');  axs[2,0].set_title('Level Thresholding '       + str(thrshlds['lvl_thr']))        
        axs[2,1].imshow(mag_thr           , cmap ='gray');  axs[2,1].set_title('magnitude thresholding: '  + str(thrshlds['mag_thr']))
        axs[3,0].imshow(dir_thr           , cmap ='gray');  axs[2,1].set_title('Directional thresholding: '+ str(thrshlds['dir_thr']))
        axs[3,1].imshow(results['cmb_mag'], cmap ='gray');  axs[3,1].set_title('Mag and Dir thresholding: '+ str(thrshlds['mag_thr']))
        axs[4,0].imshow(results['cmb_hue'], cmap ='gray');  axs[3,0].set_title('Hue thresholding: '        + str(thrshlds['hue_thr']))
        axs[4,1].imshow(results['cmb_mag_xy'], cmap ='gray');  axs[3,0].set_title('cmb_mag_xy:  (mag | xy)'+ str(thrshlds['mag_thr']))
        plt.show()

    if debug:
        fig, axs = plt.subplots(7,2,figsize=(25,49))
        axs[0,0].imshow(results['cmb_x']                , cmap ='gray');  axs[0,0].set_title('cmb_x:     (X):'+str(thrshlds['x_thr'] ))
        axs[0,1].imshow(results['cmb_mag_x']            , cmap ='gray');  axs[1,0].set_title('cmb_mag_x: (Mag/Dir)'+str(thrshlds['mag_thr'])+ ' x: '+str(thrshlds['x_thr'] ))
        axs[1,0].imshow(results['cmb_mag_lvl_x']        , cmap ='gray');  axs[1,1].set_title('cmb_mag_lvl_x:          (Mag/Dir) | Lvl | X)  image')
        axs[1,1].imshow(results['cmb_mag_sat_lvl_x']    , cmap ='gray');  axs[1,1].set_title('cmb_mag_sat_lvl_x:      (Mag/Dir) | Sat | Lvl | X)  image')
        axs[2,0].imshow(results['cmb_rgb']              , cmap ='gray');  axs[2,0].set_title('cmb_rgb:                (RGB)  '+str(thrshlds['rgb_thr']))
        axs[2,1].imshow(results['cmb_lvl']              , cmap ='gray');  axs[2,1].set_title('cmb_lvl:                (Lvl)  '+str(thrshlds['lvl_thr']))
        axs[3,0].imshow(results['cmb_sat']              , cmap ='gray');  axs[3,0].set_title('cmb_sat:                (Sat)  '+str(thrshlds['sat_thr']))
        axs[3,1].imshow(results['cmb_hue']              , cmap ='gray');  axs[3,1].set_title('cmb_hue:                (Hue)  '+str(thrshlds['hue_thr']))
        axs[4,0].imshow(results['cmb_rgb_lvl_sat']      , cmap ='gray');  axs[4,0].set_title('cmb_rgb_lvl_sat:        (RGB | Lvl | Sat)  image')
        axs[4,1].imshow(results['cmb_rgb_lvl_sat_mag']  , cmap ='gray');  axs[4,1].set_title('cmb_rgb_lvl_sat_mag:    (RGB | Lvl | Sat | (Mag/Dir)  image')
        axs[5,0].imshow(results['cmb_rgb_lvl_sat_mag_x'], cmap ='gray');  axs[5,0].set_title('cmb_rgb_lvl_sat_mag_x:  (RGB | Lvl | Sat | X |  (Mag/Dir)  image')
        axs[5,1].imshow(results['cmb_sat_mag_x']        , cmap ='gray');  axs[5,1].set_title('cmb_sat_mag_x:          (Sat | (Mag/Dir) | X  image')
        axs[6,0].imshow(results['cmb_hue_x']            , cmap ='gray');  axs[6,0].set_title('cmb_hue_x:              ( X | Hue)  image')
        axs[6,1].imshow(results['cmb_hue_mag_x']        , cmap ='gray');  axs[6,1].set_title('cmb_hue_mag_x:          ( X | Hue | (Mag/Dir))  image')
        plt.show()

    if ret is not None:
        return results[ret]    
    else:
        return results
        
def apply_perspective_transform(inputs, itStr, source, dest, **kwargs ):
    debug   = kwargs.get('debug' ,  False)
    debug2  = kwargs.get('debug2',  False)
    SIZE    = kwargs.get('size'  , (18,6))
    results = {}
    inp_ratio = {}
    res_ratio = {}
    keys      = list(inputs.keys())
    ttl_pixels  = inputs[keys[0]].shape[0] * inputs[keys[0]].shape[1]
    half_pixels = ttl_pixels // 2
    warped_p    = 'warped    %'

    for k in keys:
        results[k], _, Minv = perspectiveTransform(inputs[k] , source, dest, debug = debug2)
        inp_ratio[k] = str(round(np.sum(inputs[k][450:680,:]) * 100/half_pixels,2))
        res_ratio[k] = str(round(np.sum(results[k]) * 100/ttl_pixels,2))
        # if debug:
            # print( ' {:20s}     {:6d}      {:8.2f} '.format(k, np.count_nonzero(results[k]), res_ratio[k]))
    
    if debug:
        display_multi(inputs['cmb_x']   , results['cmb_x']  , 
                      inputs['cmb_hue'] , results['cmb_hue'], 
                      title1 = 'cmb_x   '+itStr['x_thr']  +'    %'+inp_ratio['cmb_x'], 
                      title3 = 'cmb_hue '+itStr['hue_thr']+'    %'+inp_ratio['cmb_hue'], 
                      title2 = warped_p + res_ratio['cmb_x'], 
                      title4 = warped_p + res_ratio['cmb_hue'])

        display_multi(inputs['cmb_mag'] , results['cmb_mag'], 
                      inputs['cmb_lvl'] , results['cmb_lvl'], 
                      title1 = 'cmb_mag '+itStr['mag_thr']+'    %'+inp_ratio['cmb_mag'],
                      title3 = 'cmb_lvl '+itStr['lvl_thr']+'    %'+inp_ratio['cmb_lvl'],  
                      title2 = warped_p + res_ratio['cmb_mag'], 
                      title4 = warped_p + res_ratio['cmb_lvl'])

        if itStr['mag_thr'] :
            display_multi(inputs['cmb_mag_x']       , results['cmb_mag_x'],   
                            inputs['cmb_mag_lvl_x'] , results['cmb_mag_lvl_x'],
                            title1 = 'cmb_mag_x'    +'    %'+inp_ratio['cmb_mag_x'],  
                            title3 = 'cmb_mag_lvl_x'+'    %'+inp_ratio['cmb_mag_lvl_x'],
                            title2 = warped_p + res_ratio['cmb_mag_x'], 
                            title4 = warped_p + res_ratio['cmb_mag_lvl_x'])

        if itStr['hue_thr'] != 'None': 
            display_multi(inputs['cmb_hue_x']     , results['cmb_hue_x'], 
                          inputs['cmb_hue_mag_x'] , results['cmb_hue_mag_x'], 
                          title1 = 'cmb_hue_x'    +'    %'+inp_ratio['cmb_hue_x'],  
                          title3 = 'cmb_hue_mag_x'+'    %'+inp_ratio['cmb_hue_mag_x'],
                          title2 = warped_p + res_ratio['cmb_hue_x'], 
                          title4 = warped_p + res_ratio['cmb_hue_mag_x'])

        if itStr['lvl_thr'] != 'None': 
            display_multi(inputs['cmb_hue_lvl_x']     , results['cmb_hue_lvl_x']    , 
                          inputs['cmb_hue_mag_lvl_x'] , results['cmb_hue_mag_lvl_x'], 
                          title1 = 'cmb_hue_lvl_x'    +'    %'+inp_ratio['cmb_hue_lvl_x'], 
                          title3 = 'cmb_hue_mag_lvl_x'+'    %'+inp_ratio['cmb_hue_mag_lvl_x'], 
                          title2 = warped_p + res_ratio['cmb_hue_lvl_x'], 
                          title4 = warped_p + res_ratio['cmb_hue_mag_lvl_x'])


        
        if itStr['y_thr'] != 'None':
            display_multi(inputs['cmb_y']  , results['cmb_y']  , 
                          inputs['cmb_xy'] , results['cmb_xy'] , 
                          title1 = 'cmb_y '+itStr['y_thr']+'    %'+inp_ratio['cmb_y'],  
                          title3 = 'cmb_xy'+'    %'+inp_ratio['cmb_xy'], 
                          title2 = warped_p + res_ratio['cmb_y'], 
                          title4 = warped_p + res_ratio['cmb_xy'])
        
        if itStr['rgb_thr']  != 'None' or itStr['sat_thr'] != 'None':
            display_multi(inputs['cmb_rgb'], results['cmb_rgb'], 
                          inputs['cmb_sat'], results['cmb_sat'], 
                          title1 = 'cmb_rgb '+itStr['rgb_thr']+'    %'+inp_ratio['cmb_rgb'],  
                          title3 = 'cmb_sat '+itStr['sat_thr']+'    %'+inp_ratio['cmb_sat'], 
                          title2 = warped_p + res_ratio['cmb_rgb'], 
                          title4 = warped_p + res_ratio['cmb_sat'])

        if itStr['sat_thr'] != 'None':
            display_multi(inputs['cmb_sat_x']    , results['cmb_sat_x']    ,
                          inputs['cmb_sat_mag_x'], results['cmb_sat_mag_x'],
                          title1 = 'cmb_sat_x'    +'    %'+inp_ratio['cmb_sat_x'],
                          title3 = 'cmb_sat_mag_x'+'    %'+inp_ratio['cmb_sat_mag_x'],
                          title2 = warped_p + res_ratio['cmb_sat_x'], 
                          title4 = warped_p + res_ratio['cmb_sat_mag_x'])

            display_multi(inputs['cmb_hue_mag_sat']     , results['cmb_hue_mag_sat']    , 
                          inputs['cmb_hue_mag_lvl_sat'] , results['cmb_hue_mag_lvl_sat'], 
                          title1 = 'cmb_hue_mag_sat'    +'  %'+inp_ratio['cmb_hue_mag_sat'], 
                          title3 = 'cmb_hue_mag_lvl_sat'+'  %'+inp_ratio['cmb_hue_mag_lvl_sat'], 
                          title2 = warped_p + res_ratio['cmb_hue_mag_sat'], 
                          title4 = warped_p + res_ratio['cmb_hue_mag_lvl_sat'])


        if itStr['rgb_thr'] != 'None' or itStr['lvl_thr'] != 'None':
            display_multi(inputs['cmb_rgb_lvl']    , results['cmb_rgb_lvl'] ,          
                          inputs['cmb_rgb_lvl_sat'], results['cmb_rgb_lvl_sat'] , 
                          title1 = 'cmb_rgb_lvl'    +'    %'+inp_ratio['cmb_rgb_lvl'], 
                          title3 = 'cmb_rgb_lvl_sat'+'    %'+inp_ratio['cmb_rgb_lvl_sat'], 
                          title2 = warped_p + res_ratio['cmb_rgb_lvl'], 
                          title4 = warped_p + res_ratio['cmb_rgb_lvl_sat'])

            
            display_multi(inputs['cmb_rgb_lvl_sat_mag']  , results['cmb_rgb_lvl_sat_mag'],   
                          inputs['cmb_rgb_lvl_sat_mag_x'], results['cmb_rgb_lvl_sat_mag_x'], 
                          title1 = 'cmb_rgb_lvl_sat_mag'  +'    %'+inp_ratio['cmb_rgb_lvl_sat_mag'], 
                          title3 = 'cmb_rgb_lvl_sat_mag_x'+'    %'+inp_ratio['cmb_rgb_lvl_sat_mag_x'],
                          title2 = warped_p + res_ratio['cmb_rgb_lvl_sat_mag'], 
                          title4 = warped_p + res_ratio['cmb_rgb_lvl_sat_mag_x'])


    return  results


def apply_thresholds_v1(img, ret = None, ksize = 10, x_thr = 0, y_thr = 0, mag_thr = 0, dir_thr = 0, sat_thr = None,
                          lvl_thr = None, rgb_thr = None, debug = False):
       
    # Apply each of the thresholding functions
    x_binary   = grad_abs_thresh(img, orient='x', sobel_kernel=ksize, thresh=x_thr)

    if y_thr is None :
        y_binary = np.ones_like(x_binary)
    else:
        y_binary = grad_abs_thresh(img, orient='y', sobel_kernel=ksize, thresh=y_thr)
    
    if rgb_thr is None:
        rgb_binary = np.zeros_like(x_binary)
    else:
        rgb_binary = RGB_thresh(img, thresh = rgb_thr)
        
    if lvl_thr is None:
        lvl_binary = np.zeros_like(x_binary)
    else:
        lvl_binary = level_thresh(img, thresh = lvl_thr)
        
    if sat_thr is None:
        sat_binary = np.zeros_like(x_binary)
    else:
        sat_binary = saturation_thresh(img, thresh = sat_thr)
    
    mag_binary = grad_mag_thresh(img, sobel_kernel=ksize, thresh=mag_thr)
    dir_binary = grad_dir_thresh(img, sobel_kernel=ksize, thresh=dir_thr)
        
    results    = {}
    
    # cmb_x_sat  = np.zeros_like(dir_binary)
    # cmb_x_sat[(x_binary == 1) | (sat_binary ==1)] = 1
    # results['cmb_x_sat'] = cmb_mag_dir

    # cmb_x_sat_mag  = np.zeros_like(dir_binary)
    # cmb_x_sat_mag[(x_binary == 1) | (cmb_mag_dir == 1) | (sat_binary ==1)] = 1
    # results['cmb_x_sat_mag'] = cmb_x_sat_mag

    # cmb_x_sat_mag_rgb  = np.zeros_like(dir_binary)
    # cmb_x_sat_mag_rgb[(x_binary == 1)  | (cmb_mag_dir == 1) | (sat_binary ==1) | (rgb_or_binary == 1)] = 1
    # results['cmb_x_sat_mag_rgb'] = cmb_x_sat_mag_rgb

    cmb_xy  = np.zeros_like(dir_binary)
    cmb_xy[ ((x_binary == 1) & (y_binary == 1)) ] = 1
    results['cmb_xy'] = cmb_xy

    cmb_mag_dir  = np.zeros_like(dir_binary)
    cmb_mag_dir[ ((mag_binary == 1) & (dir_binary == 1)) ] = 1
    # results['cmb_mag_dir'] = cmb_mag_dir

    cmb_rgb_lvl = np.zeros_like(dir_binary)
    cmb_rgb_lvl[ ((rgb_binary == 1) | (lvl_binary == 1)) ] = 1
    results['cmb_rgb_lvl'] = cmb_rgb_lvl

    cmb_rgb_lvl_sat = np.zeros_like(dir_binary)
    cmb_rgb_lvl_sat[ ((cmb_rgb_lvl == 1) | (sat_binary == 1)) ] = 1
    results['cmb_rgb_lvl_sat'] = cmb_rgb_lvl_sat

    cmb_rgb_lvl_sat_mag = np.zeros_like(dir_binary)
    cmb_rgb_lvl_sat_mag[ (cmb_mag_dir  == 1) | ((cmb_rgb_lvl == 1) | (sat_binary == 1)) ] = 1
    results['cmb_rgb_lvl_sat_mag'] = cmb_rgb_lvl_sat_mag

    cmb_rgb_lvl_sat_mag_x = np.zeros_like(dir_binary)
    cmb_rgb_lvl_sat_mag_x[ (cmb_mag_dir  == 1) | ((cmb_rgb_lvl == 1) | (sat_binary == 1) | (x_binary == 1)) ] = 1
    results['cmb_rgb_lvl_sat_mag_x'] = cmb_rgb_lvl_sat_mag_x

    cmb_mag_x = np.zeros_like(dir_binary)
    cmb_mag_x[ (cmb_mag_dir  == 1) | (x_binary == 1) ] = 1
    results['cmb_mag_x'] = cmb_mag_x
    
    results['cmb_sat'] = sat_binary
    results['cmb_lvl'] = lvl_binary
    
    
    if debug:
        fig = plt.figure(figsize=(25,28))
        plt.subplot(421);plt.imshow(img); plt.title(' input image: '+str(img.shape[0])+' x '+str(img.shape[1]))
        plt.subplot(422);plt.imshow(x_binary   , cmap ='gray'); plt.title('grad x thresholding: '+str(x_thr))
        plt.subplot(423);plt.imshow(y_binary   , cmap ='gray'); plt.title('grad y thresholding: '+str(y_thr)) 
        plt.subplot(424);plt.imshow(mag_binary , cmap ='gray'); plt.title('magnitude thresholding: '+str(mag_thr))
        plt.subplot(425);plt.imshow(dir_binary , cmap ='gray'); plt.title('directional thresholding: '+str(dir_thr))
        plt.subplot(426);plt.imshow(cmb_mag_dir, cmap ='gray'); plt.title('cmb_mag_dir (Dir & Mag) thresholding')
        plt.subplot(427);plt.imshow(sat_binary , cmap ='gray'); plt.title('Saturation thresholding: '+str(sat_thr)+' ')
        plt.subplot(428);plt.imshow(lvl_binary , cmap ='gray'); plt.title('Level Thresholding: '+str(lvl_thr)+' ')        
        plt.show()

        # plt.subplot(424);plt.imshow(cmb_xy    , cmap ='gray'); plt.title(' cmb_xy:  (X/Y)  x: '+str(x_thr)+' y: '+str(y_thr))
        # plt.subplot(424);plt.imshow(cmb_x_sat, cmap ='gray'); plt.title('cmb_x_sat: (X | Sat) image')
        # plt.subplot(425);plt.imshow(cmb_x_sat_mag       , cmap ='gray'); plt.title('cmb_x_sat_mag: (X | Sat | Dir/Mag) image')
        # plt.subplot(426);plt.imshow(cmb_x_sat_mag_rgb   , cmap ='gray'); plt.title('cmb_x_sat_mag_rgb: (X | Sat | Dir/Mag | RGB) image')
        
        fig = plt.figure(figsize=(25,28))
        plt.subplot(421);plt.imshow(rgb_binary            , cmap ='gray'); plt.title('RGB OR Thresholding: '+str(rgb_thr)+' ')
        plt.subplot(422);plt.imshow(cmb_rgb_lvl           , cmap ='gray'); plt.title('cmb_rgb_lvl:  (RGB | Lvl)  image')
        plt.subplot(423);plt.imshow(cmb_rgb_lvl_sat       , cmap ='gray'); plt.title('cmb_rgb_lvl_sat:  (RGB | Lvl | Sat)  image')
        plt.subplot(424);plt.imshow(cmb_rgb_lvl_sat_mag   , cmap ='gray'); plt.title('cmb_rgb_lvl_sat_mag:  (RGB | Lvl | Sat) AND (Mag/Dir)  image')
        plt.subplot(425);plt.imshow(cmb_rgb_lvl_sat_mag_x , cmap ='gray'); plt.title('cmb_rgb_lvl_sat_mag_x:  (RGB | Lvl | Sat | X) AND (Mag/Dir)  image')
        plt.subplot(426);plt.imshow(cmb_mag_x             , cmap ='gray'); plt.title('cmb_mag_x:   X:'+str(x_thr)+' (Mag/Dir) '+str(mag_thr))
        plt.show()

    if ret is not None:
        return results[ret]    
    else:
        return results

"""    
def RGB_AND_thresh(img, thresh=(0, 255),  display = False):
    s_channel = np.copy(img)
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
  
    binary_output = np.max(binary_output, axis = -1)

    if display:
        
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        # filtered_dir = s_channel * np.dstack((binary_output,binary_output,binary_output))
   
        print(' filtered_dir : ', filtered_dir.shape)
        print(' s_channel    : ', s_channel.shape)
        print(' binary_output: ', binary_output.shape)
        display_one(s_channel, title='RGB channels   Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) )    
        display_one(filtered_dir , title='filtered_grad - within thresholds: '+thresh_str+' Min: ' +str(filtered_dir.min())+ ' Max: '+str(filtered_dir.max()) )        
        print(binary_output.shape, binary_output.min(), binary_output.max())
        display_one(binary_output, title='Thresholded Output - thresholds: '+thresh_str)    
    
    return binary_output
"""

    
