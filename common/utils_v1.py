import sys,os, pprint
if '..' not in sys.path:
    print("utils.py: appending '..' to sys.path")
    sys.path.append('..')

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from   matplotlib import colors
import pickle
import pprint 
import collections
pp = pprint.PrettyPrinter(indent=2, width=100)

  
# from classes.line import Line ## import classes.line as line
import classes.line
RED   = 0
GREEN = 1
BLUE  = 2


NWINDOWS = 9
MINPIX = 90
MAXPIX = 8000
WINDOW_SEARCH_MARGIN = 100
POLY_SEARCH_MARGIN   = 100



def colorLanePixels_v1(input_img, leftx, lefty, rightx, righty, lcolor = 'red', rcolor = 'blue', debug = False):
    if input_img.shape[-1] == 4:
        color_left  = colors.to_rgba_array(lcolor)[0]* 255
        color_right = colors.to_rgba_array(rcolor)[0]* 255
    else:
        color_left  = colors.to_rgb_array(lcolor)[0]* 255
        color_right = colors.to_rgb_array(rcolor)[0]* 255
                      
                      

    ## Visualization ##
    # Colors in the left and right lane regions
    output_img = np.copy(input_img)
    output_img[lefty, leftx]   = color_left ## [255, 0, 0]
    output_img[righty, rightx] = color_right ## [0, 0, 255]
    
    return output_img


def fit_polynomial_v1(leftx, lefty, rightx, righty, debug = False):
    ### Fit a second order polynomial to each using `np.polyfit` ###
    left_fit  = np.polyfit(lefty , leftx , 2, full=False)
    right_fit = np.polyfit(righty, rightx, 2, full=False)

    if debug: 
        print('\nfit_polynomial:')
        print('-'*20)    
        print(' left poly coeffs :', left_fit)
        print(' right poly coeffs:', right_fit)

    return left_fit, right_fit
    
def plot_polynomial_v1(height, left_fit, right_fit, debug = False):
    # Generate y values for plotting
    ploty = np.linspace(0, height-1, height)

    try:
        left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx  = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
    return  ploty, left_fitx, right_fitx    

def displayLaneRegion_v1(input_img, left_fitx, right_fitx,  Minv, **kwargs ):
    ''' 
    
    '''
    beta  = kwargs.setdefault( 'beta', 0.5) 
    start = kwargs.setdefault('start', 0)  
    debug = kwargs.setdefault('debug', False)
    
    left_height  = left_fitx.shape[0]
    right_height = right_fitx.shape[0]    
    
    left_ploty   = (np.linspace(start, left_height-1 , left_height , dtype = np.int))
    right_ploty  = (np.linspace(start, right_height-1, right_height, dtype = np.int))

    # Create an image to draw the lines on
    color_warp = np.zeros_like(input_img).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, left_ploty]))]).astype(np.int)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_ploty])))]).astype(np.int)
    pts       = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, ([pts]), (0,255, 0))
    
    # draw left and right lanes
    cv2.polylines(color_warp, (pts_left) , False, (255,0,0), thickness=18, lineType = cv2.LINE_AA)
    cv2.polylines(color_warp, (pts_right), False, (0,0,255), thickness=18, lineType = cv2.LINE_AA)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (input_img.shape[1], input_img.shape[0])) 
    

    if debug:
        print('undistImage : ', input_img.shape, ' newwarp : ', newwarp.shape)
    
    # Combine the result with the original image
    result = cv2.addWeighted(input_img, 1, newwarp, beta, 0)
    
    return result        
    
    
def offCenterMsg_v1(y_eval, left_fitx, right_fitx, center_x, units = 'm', debug = False):
    '''
    Calculate position of vehicle relative to center of lane
    y_eval:                 y-value where we want radius of curvature
    left_fitx, right_fitx:  x_values at y_eval
    center_x:               center of image, represents center of vehicle
    units   :               units to calculate off_center distance in 
                            pixels 'p'  or meters 'm'
    '''
    
    mid_point  = left_fitx + (right_fitx - left_fitx)//2
    off_center_pxls = mid_point - center_x 
    off_center_mtrs = off_center_pxls * (3.7/700)
    
    
    if debug: 
        print('Y: {:4.0f}  Left lane: {:8.3f}  right_lane: {:8.3f}  midpt: {:8.3f}  off_center: {:8.3f} , off_center(mtrs): {:8.3f} '.format(
                y_eval, left_fitx, right_fitx, mid_point, off_center_pxls, off_center_mtrs))
    
    oc = off_center_mtrs if units == 'm' else off_center_pxls
    
    if off_center_pxls != 0 :
        output = str(abs(round(oc,3)))+(' m ' if units == 'm' else ' pxls ')  +('left' if oc > 0 else 'right')+' of lane center'
    else:
        output = 'On lane center'
    
    return output ## round(off_center_mtrs,3), round(off_center,3)


def find_lane_pixels_v1(binary_warped, histRange = None,
                        nwindows = 9, 
                        window_margin = 100, 
                        minpix   = 50,
                        maxpix   = 99999, 
                        debug    = False):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    # nwindows = 9
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Set the width of the windows +/- margin
    # margin = 100
    # Set minimum number of pixels found to recenter window
    # minpix = 90
    # Set maximum number of pixels found to recenter window
    if maxpix == 0 :
        maxpix = (window_height * window_margin)
        
        
    # LLane.set_height(binary_warped.shape[0])
    # RLane.set_height(binary_warped.shape[0])

    if histRange is None:
        histLeft = 0
        histRight = binary_warped.shape[1]
    else:
        histLeft, histRight = int(histRange[0]), int(histRange[1])

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped , binary_warped, binary_warped,  np.ones_like(binary_warped)))
    out_img *= 255
    
    
    # Take a histogram of the bottom half of the image    
    histogram = np.sum(binary_warped[2*binary_warped.shape[0]//3:, histLeft:histRight], axis=0)
    if debug:
        display_two(binary_warped, out_img, title1 = 'binary_warped '+str(binary_warped.shape))
        print(' histogram shape before padding: ' , histogram.shape)
    
    histogram = np.pad(histogram, (histLeft, binary_warped.shape[1]-histRight))
    if debug:
        print(' histogram shape after padding : ' , histogram.shape) 
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint    = np.int(histogram.shape[0]//2)
    leftx_base  = np.argmax(histogram[:midpoint]) 
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint   

    
    if debug:
        print(' Run find_lane_pixels()  - histRange:', histRange)
        print(' Midpoint (histogram//2): {} '.format(midpoint))
        print(' Histogram left side max: {}  right side max: {}'.format(np.argmax(histogram[:midpoint]),np.argmax(histogram[midpoint:])))
        print(' Histogram left side max: {}  right side max: {}'.format(leftx_base, rightx_base))
    
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
  
    # Current positions to be updated later for each window in nwindows
    leftx_current  = leftx_base
    rightx_current = rightx_base
    
    if debug:
        print(' left x base       : ', leftx_base, '  right x base :', rightx_base )
        print(' window_height     : ', window_height)
        print(' nonzero x count   : ', nonzerox.shape[0])
        print(' nonzero y count   : ', nonzeroy.shape[0])
        print(' Starting Positions: left x :', leftx_current, '  right x: ', rightx_current )
    
    # Create empty lists to receive left and right lane pixel indices
    left_line_inds  = []
    right_line_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low   = leftx_current  - window_margin  
        win_xleft_high  = leftx_current  + window_margin  
        win_xright_low  = rightx_current - window_margin       
        win_xright_high = rightx_current + window_margin  

        if debug:
            print()
            print(' Window: ', window, ' y range: ', win_y_low,' to ', win_y_high )
            print('-'*50)
            print(' Left  lane X range : ', win_xleft_low , '  to  ', win_xleft_high)
            print(' Right lane X range : ', win_xright_low, '  to  ', win_xright_high)
            
        # Draw the windows on the visualization image
        window_color = colors.to_rgba_array('green')[0]* 255

        cv2.rectangle(out_img,(win_xleft_low , win_y_low), (win_xleft_high , win_y_high), window_color, 2) 
        cv2.rectangle(out_img,(win_xright_low, win_y_low), (win_xright_high, win_y_high), window_color, 2) 
        
        ### MY SOLUTION: Identify the nonzero pixels in x and y within the window -------------
        left_x_inds = np.where((win_xleft_low <=  nonzerox) & (nonzerox < win_xleft_high))
        left_y_inds = np.where((win_y_low     <=  nonzeroy) & (nonzeroy < win_y_high))
        good_left_inds = np.intersect1d(left_x_inds,left_y_inds,assume_unique=False)
        
        right_x_inds = np.where((win_xright_low <= nonzerox) & (nonzerox < win_xright_high))
        right_y_inds = np.where((win_y_low     <=  nonzeroy) & (nonzeroy < win_y_high))
        good_right_inds = np.intersect1d(right_x_inds,right_y_inds,assume_unique=False)
        ###------------------------------------------------------------------------------------

        ### UDACITY SOLUTION: Identify the nonzero pixels in x and y within the window ###
        # good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        # (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        # good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        # (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        ###------------------------------------------------------------------------------------
        
        if debug:
            print()
            print(' left_x_inds  : ', left_x_inds[0].shape, ' left_y_indx  : ', left_y_inds[0].shape,
                  ' -- good left inds size: ', good_left_inds.shape[0])
            # print(' Avg: ', int(nonzerox[good_left_inds].mean()))
            # print(' percentile :', np.percentile(nonzerox[good_left_inds], [25,50,75]))
            # print(' X: ', nonzerox[good_left_inds]) ; print(' Y: ', nonzeroy[good_left_inds])
            print(' right_x_inds : ', right_x_inds[0].shape, ' right_y_indx : ', right_y_inds[0].shape,
                  '  -- good right inds size: ', good_right_inds.shape[0])
            # print(' Avg: ', int(nonzerox[good_right_inds].mean()))
            # print(' percentile :', np.percentile(nonzerox[good_right_inds], [25,50,75]))
            # print(' X: ', nonzerox[good_right_inds]); print(' Y: ', nonzeroy[good_right_inds])
        
        # Append these indices to the lists
        left_line_inds.append(good_left_inds)
        right_line_inds.append(good_right_inds)
        
        ### If #pixels found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position    ###
        if (maxpix > good_left_inds.shape[0] > minpix):
            left_msg  = ' Set leftx_current  :  {} ---> {} '.format( leftx_current,  int(nonzerox[good_left_inds].mean()))
            leftx_current = int(nonzerox[good_left_inds].mean())
        else:
            left_msg  = ' Keep leftx_current :  {} '.format(leftx_current)

        if (maxpix > good_right_inds.shape[0] > minpix ) :
            right_msg = ' Set rightx_current :  {} ---> {} '.format( rightx_current, int(nonzerox[good_right_inds].mean()))
            rightx_current = int(nonzerox[good_right_inds].mean())
        else:
            right_msg = ' Keep rightx_current:  {} '.format(rightx_current)
        
        if debug:
            print(left_msg)
            print(right_msg)

        
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_line_inds  = np.concatenate(left_line_inds)
        right_line_inds = np.concatenate(right_line_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print(' concatenate not working ')
        rc = 0 
    else:
        # Extract left and right line pixel positions
        leftx  = nonzerox[left_line_inds] ; lefty  =  nonzeroy[left_line_inds]
        rightx = nonzerox[right_line_inds]; righty = nonzeroy[right_line_inds]
        rc = 1
        
    if debug:
        print()
        print(' leftx : ', leftx.shape, ' lefty : ', lefty.shape)
        print(' rightx : ', rightx.shape, ' righty : ', righty.shape)

    return leftx, lefty, rightx, righty, out_img, histogram
   
