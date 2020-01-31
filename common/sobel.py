import numpy as np
import cv2
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from .utils import get_ax, display_one_cbar, display_arctan, display_arctan2
deg2rad = lambda  x: x * np.pi/180
rad2deg = lambda  x: x * 180 / np.pi

def convert_to_gray(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.copy(img)
    return gray
    
    


# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=20, thresh_max=100
# should produce output like the example image shown above this quiz.
def grad_abs_thresh(img, orient='x', sobel_kernel=3 , thresh=(0, 255), display = False):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    gray = convert_to_gray(img)
    
    x_sobel = 1 if orient == 'x' else 0
    y_sobel = 1 if x_sobel == 0 else 0
    # print(x_sobel, y_sobel)
    sobel = cv2.Sobel(gray, cv2.CV_64F, x_sobel, y_sobel, ksize = sobel_kernel)
        
    abs_sobel = np.absolute(sobel)
    
    scaled_sobel = np.uint8(255 * abs_sobel / abs_sobel.max())
    
    
    # print('sobel       : ', sobel.shape, sobel.min(), sobel.max())
    # print('scaled_sobel: ', scaled_sobel.shape, scaled_sobel.min(), scaled_sobel.max())
    # print('abs_sobel   : ', abs_sobel.shape, abs_sobel.min(), abs_sobel.max())
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    
    if display:
        print('grad_abs_thresh(): input:', img.shape, ' min: ', img.min(), ' max: ', img.max())
        filtered_sobel = np.copy(scaled_sobel)
        filtered_sobel[(scaled_sobel < thresh[0]) | (scaled_sobel > thresh[1])] = 0
        display_one_cbar(scaled_sobel , title='scaled_sobel - orientation: '+orient+'  thresholds: '+str(thresh), cmap='jet')
        display_one_cbar(filtered_sobel, title='filtered_sobel - orientation: '+orient+' within thresholds: '+str(thresh), cmap='jet')
        display_one_cbar(binary_output, title='result image - orientation: '+orient+'  thresholds: '+str(thresh))
    return binary_output


def grad_mag_thresh(img, sobel_kernel=3, thresh=(0, 255), display = False):
    '''
    Define a function that applies Sobel x and y, 
    then computes the magnitude of the gradient
    and applies a threshold
    
    Apply the following steps to img
     1) Convert to grayscale
     2) Take the gradient in x and y separately
     3) Calculate the magnitude 
     4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
     5) Create a binary mask where mag thresholds are met
     6) Return this mask as your binary_output image
    ''' 
    
    thresh_min, thresh_max = thresh
    
    # 1) Convert to grayscale
    gray = convert_to_gray(img)
    
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)    
    
    # 3) Calculate the magnitude 
    abs_sobel = np.sqrt((np.absolute(sobel_x )** 2) + (np.absolute(sobel_y) ** 2))
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / abs_sobel.max())
    
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    
    if display:
        filtered_sobel = np.copy(scaled_sobel)
        filtered_sobel[(scaled_sobel < thresh[0]) | (scaled_sobel > thresh[1])] = 0
        display_one_cbar(scaled_sobel  , title='scaled_sobel - thresholds: '+str(thresh), cmap='jet')
        display_one_cbar(filtered_sobel, title='filtered_sobel - within thresholds: '+str(thresh), cmap='jet')
        display_one_cbar(binary_output , title='result image - thresholds: '+str(thresh))    
    return binary_output

def grad_dir_thresh(img, sobel_kernel=3, thresh=(0,90), display = False):
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
    theta1, theta2  = thresh
    thresh_radians = (deg2rad(theta1), deg2rad(theta2))
    
    gray = convert_to_gray(img)
    
    abs_sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    abs_sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    
    grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
    grad_dir_deg = np.rad2deg(grad_dir)
    
    cmin = grad_dir.min(); cmax = grad_dir.max()
    cmin_deg = grad_dir_deg.min(); cmax_deg = grad_dir_deg.max()
    
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh_radians [0]) & (grad_dir <= thresh_radians [1])] = 1
    
    if display:
        print("grad_dir:", grad_dir.shape, ' Deg: ', str(thresh) + ' Rad: ' + str(thresh_radians))
        filtered_dir = np.copy(grad_dir)
        filtered_dir[(grad_dir < thresh_radians[0]) | (grad_dir > thresh_radians[1])] = 0
        filtered_dir_deg = np.copy(grad_dir_deg)
        filtered_dir_deg[(grad_dir_deg < thresh[0]) | (grad_dir_deg > thresh[1])] = 0

        
        display_arctan(grad_dir        , title='gradient direction (Radians)' )
        display_arctan(grad_dir_deg    , title='gradient direction (Degrees)' )
        display_arctan(filtered_dir    , title='filtered_grad - w/i thresholds Deg: '+str(thresh)+ 'Rad:'+str(thresh_radians), clim = (cmin, cmax))
        display_arctan(filtered_dir_deg, title='filtered_grad_deg - w/i thresholds Deg: '+str(thresh)+ 'Rad:'+str(thresh_radians), clim = (cmin_deg, cmax_deg))
        display_one_cbar(binary_output , title='result image - thresholds: Deg: '+str(thresh)+ 'Rad:'+str(thresh_radians))  
    return binary_output

def color_thresh(img, thresh=(0, 255), channel = 0, display = False):
    s_channel = img[:,:,channel]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        print(' filtered_dir : ', filtered_dir.shape)
        print(' s_channel    : ', s_channel.shape)
        print(' binary_output: ', binary_output.shape)
        
        display_one_cbar(s_channel, title='RGB channel '+str(channel)+ '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir, title='filtered_grad - within thresholds: '+str(thresh), cmap = 'jet' )        
        display_one_cbar(binary_output, title='Thresholded Output- thresholds: '+str(thresh))    
    
    return binary_output

def RGB_OR_thresh(img, thresh=(0, 255),  display = False):
    s_channel = np.copy(img)
    red_binary   = color_thresh(s_channel, thresh = thresh, channel = 0 ,display = display)
    green_binary = color_thresh(s_channel, thresh = thresh, channel = 1, display = display)
    blue_binary  = color_thresh(s_channel, thresh = thresh, channel = 2, display = display)
 
    binary_output = np.zeros_like(s_channel[:,:,0])
    binary_output[ (red_binary == 1) | (green_binary == 1)| (blue_binary ==1) ] = 1
  

    if display:
        
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        # filtered_dir = s_channel * np.dstack((binary_output,binary_output,binary_output))
   
        print(' filtered_dir : ', filtered_dir.shape)
        print(' s_channel    : ', s_channel.shape)
        print(' binary_output: ', binary_output.shape)
        display_one_cbar(s_channel, title='RGB channels   Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir , title='filtered_grad - within thresholds: '+str(thresh)+' Min: ' +str(filtered_dir.min())+ ' Max: '+str(filtered_dir.max()) , cmap = 'jet' )        
        print(binary_output.shape, binary_output.min(), binary_output.max())
        display_one_cbar(binary_output, title='Thresholded Output - thresholds: '+str(thresh))    
    
    return binary_output
    
"""    
def RGB_thresh(img, thresh=(0, 255),  display = False):
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
        display_one_cbar(s_channel, title='RGB channels   Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir , title='filtered_grad - within thresholds: '+str(thresh)+' Min: ' +str(filtered_dir.min())+ ' Max: '+str(filtered_dir.max()) , cmap = 'jet' )        
        print(binary_output.shape, binary_output.min(), binary_output.max())
        display_one_cbar(binary_output, title='Thresholded Output - thresholds: '+str(thresh))    
    
    return binary_output
"""

    
def saturation_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hlsImage[:,:,-1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one_cbar(s_channel, title='Saturation channel '+str(thresh) + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir, title='filtered_grad - within thresholds: '+str(thresh), cmap = 'jet' )        
        display_one_cbar(binary_output, title='Thresholded Output- thresholds: '+str(thresh))    
    
    return binary_output

def hue_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hlsImage[:,:,0]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    
    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one_cbar(s_channel, title='Hue channel '+str(thresh) + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir, title='filtered_grad - within thresholds: '+str(thresh), cmap = 'jet' )        
        display_one_cbar(binary_output, title='Thresholded Output- thresholds: '+str(thresh))    
        
    return binary_output

def level_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hlsImage[:,:,1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one_cbar(s_channel, title='Level channel '+str(thresh) + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir, title='filtered_grad - within thresholds: '+str(thresh), cmap = 'jet' )        
        display_one_cbar(binary_output, title='Thresholded Output- thresholds: '+str(thresh))    
    return binary_output
    
def HSV_value_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = hlsImage[:,:,-1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one_cbar(s_channel, title='Level channel '+str(thresh) + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir, title='filtered_grad - within thresholds: '+str(thresh), cmap = 'jet' )        
        display_one_cbar(binary_output, title='Thresholded Output- thresholds: '+str(thresh))    
    return binary_output

def YCrCb_Y_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    s_channel = hlsImage[:,:,0]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one_cbar(s_channel, title='Level channel '+str(thresh) + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir, title='filtered_grad - within thresholds: '+str(thresh), cmap = 'jet' )        
        display_one_cbar(binary_output, title='Thresholded Output- thresholds: '+str(thresh))    
    return binary_output

def YCrCb_Cr_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    s_channel = hlsImage[:,:,1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one_cbar(s_channel, title='Level channel '+str(thresh) + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir, title='filtered_grad - within thresholds: '+str(thresh), cmap = 'jet' )        
        display_one_cbar(binary_output, title='Thresholded Output- thresholds: '+str(thresh))    
    return binary_output

def YCrCb_Cb_thresh(img, thresh=(0, 255), display = False):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    s_channel = hlsImage[:,:,-1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    if display:
        filtered_dir = np.copy(s_channel)
        filtered_dir[(s_channel < thresh[0]) | (s_channel > thresh[1])] = 0
        
        display_one_cbar(s_channel, title='Level channel '+str(thresh) + '  Min: ' +str(s_channel.min())+ ' Max: '+str(s_channel.max()) , cmap = 'jet')    
        display_one_cbar(filtered_dir, title='filtered_grad - within thresholds: '+str(thresh), cmap = 'jet' )        
        display_one_cbar(binary_output, title='Thresholded Output- thresholds: '+str(thresh))    
    return binary_output

def apply_thresholds(img, ret = None, ksize = 10, x_thr = 0, y_thr = 0, mag_thr = 0, dir_thr = 0, sat_thr = 0,
                          lvl_thr = None, rgb_thr = None, debug = False):
       
    # Apply each of the thresholding functions
    x_binary   = grad_abs_thresh(img, orient='x', sobel_kernel=ksize, thresh=x_thr)
    if y_thr == 0:
        y_binary = np.ones_like(x_binary)
    else:
        y_binary = grad_abs_thresh(img, orient='y', sobel_kernel=ksize, thresh=y_thr)
    
    # if rgb_thr is None:
        # rgb_binary = np.zeros_like(x_binary)
    # else:
        # rgb_binary = RGB_thresh(img, thresh = rgb_thr)

    if rgb_thr is None:
        rgb_binary = np.zeros_like(x_binary)
    else:
        rgb_binary = RGB_OR_thresh(img, thresh = rgb_thr)
        
    if lvl_thr is None:
        lvl_binary = np.zeros_like(x_binary)
    else:
        lvl_binary = level_thresh(img, thresh = lvl_thr)
        
    mag_binary = grad_mag_thresh(img, sobel_kernel=ksize, thresh=mag_thr)
    dir_binary = grad_dir_thresh(img, sobel_kernel=ksize, thresh=dir_thr)
    sat_binary   = saturation_thresh(img, thresh = sat_thr)
    
    
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
        plt.subplot(427);plt.imshow(sat_binary , cmap ='gray'); plt.title('Saturation thresholding '+str(sat_thr)+' ')
        plt.subplot(428);plt.imshow(lvl_binary , cmap ='gray'); plt.title('Level Thresholding '+str(lvl_thr)+' ')        
        plt.show()

        # plt.subplot(424);plt.imshow(cmb_xy    , cmap ='gray'); plt.title(' cmb_xy:  (X/Y)  x: '+str(x_thr)+' y: '+str(y_thr))
        # plt.subplot(424);plt.imshow(cmb_x_sat, cmap ='gray'); plt.title('cmb_x_sat: (X | Sat) image')
        # plt.subplot(425);plt.imshow(cmb_x_sat_mag       , cmap ='gray'); plt.title('cmb_x_sat_mag: (X | Sat | Dir/Mag) image')
        # plt.subplot(426);plt.imshow(cmb_x_sat_mag_rgb   , cmap ='gray'); plt.title('cmb_x_sat_mag_rgb: (X | Sat | Dir/Mag | RGB) image')
        
        fig = plt.figure(figsize=(25,28))
        plt.subplot(421);plt.imshow(rgb_binary            , cmap ='gray'); plt.title('RGB OR Thresholding '+str(rgb_thr)+' ')
        plt.subplot(422);plt.imshow(cmb_rgb_lvl           , cmap ='gray'); plt.title('cmb_rgb_lvl:  (RGB | Lvl)  image')
        plt.subplot(423);plt.imshow(cmb_rgb_lvl_sat       , cmap ='gray'); plt.title('cmb_rgb_lvl_sat:  (RGB | Lvl | Sat)  image')
        plt.subplot(424);plt.imshow(cmb_rgb_lvl_sat_mag   , cmap ='gray'); plt.title('cmb_rgb_lvl_sat_mag:  (RGB | Lvl | Sat) AND (Mag/Dir)  image')
        plt.subplot(425);plt.imshow(cmb_rgb_lvl_sat_mag_x , cmap ='gray'); plt.title('cmb_rgb_lvl_sat_mag_x:  (RGB | Lvl | Sat | X) AND (Mag/Dir)  image')
        plt.subplot(426);plt.imshow(cmb_mag_x             , cmap ='gray'); plt.title('cmb_mag_x:  ( X | (Mag/Dir))  image')
        plt.show()

    if ret is not None:
        return results[ret]    
    else:
        return results