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

print(os.getcwd())
pp.pprint(sys.path)
  
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

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    fig , ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig,ax
    
def display_one(img, title = 'Input image ', grayscale = False,
    ax = None, size = 15, winttl= '', grid = None):
     
    cmap = 'gray' if img.ndim == 2 else 'jet'
    
    if ax is None:
        fig, ax = get_ax(rows =1, cols = 1, size= size)
    fig.canvas.set_window_title(winttl+' : '+ title)        
    
    if img.ndim == 1:
        ax.plot(img)
    else:
        ax.imshow(img, cmap = cmap)
    
    # if grayscale and img.ndim == 3:
        # imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # ax.imshow(imgGray, cmap='gray')
    # else:
        # ax.imshow(img, cmap = cmap)
    if grid is not None:
        ax.minorticks_on()
        ax.grid(True, which=grid)    
    ax.set_title(title+'  '+str(img.shape))
    plt.show()

def display_one_cbar(image, title = 'Image', cmap = 'gray', ax = None, size = 15):
    
    if ax is None:
        fig, ax = get_ax(rows =1, cols = 1, size= size)
    # fig, ax = plt.figure( figsize=(20,10));

    cs = plt.imshow(image, cmap=cmap)
    fig.colorbar(cs, shrink=0.5, aspect=20, fraction=0.05)

    if cmap == 'gray':
        plt.clim(0,1)
    else:
        plt.clim(0,255)

    ax.set_title(title)
    # plt.show()    
    return
    
def display_arctan(image, title = 'Image', cmap = 'jet' , clim = None, size = 15):
    # fig = plt.figure( figsize=(25,15));
    
    fig, ax = get_ax(rows =1, cols = 1, size= size)
    
    cs = plt.imshow(image, cmap=cmap)
    fig.colorbar(cs, shrink=0.5, aspect=20, fraction=0.05)
    if clim is not None:
        plt.clim(clim)
    plt.title(title)
    plt.show()    
    return

def display_arctan2(image, title = 'Image', cmap = 'jet' , clim = (0,np.pi/2), size = 15):
    fig, ax = get_ax(rows =1, cols = 1, size= size)
   
    cs = plt.imshow(image, cmap=cmap)
    fig.colorbar(cs, shrink=0.7, aspect=30, fraction=0.05)
    # plt.clim(clim)
    plt.title(title)
    plt.show()    
    return
    

def display_two(img1, img2, title1 = 'Original Image: ', title2 ='Undistorted Image: ', 
                size = (24,9), winttl = '', grid1 = False, grid2 = False):
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
    f.canvas.set_window_title(winttl+' : '+ title1+' - '+title2)
    f.tight_layout()
    
    cmap = 'gray' if img1.ndim == 2 else 'jet'
    if img1.ndim == 1:
        ax1.plot(img1)
    else:
        ax1.imshow(img1, cmap = cmap)
    ax1.set_title(title1+'   '+str(img1.shape), fontsize=15)
    ax1.minorticks_on()
    ax1.grid(grid1, which='both')
    # ax1.grid(True,which = 'both', axis='both', color='r', linestyle = '-', linewidth = 1)

    cmap = 'gray' if img2.ndim == 2 else 'jet'
    if img2.ndim == 1:
        ax2.plot(img2)
    else:
        ax2.imshow(img2, cmap = cmap)
    ax2.set_title(title2+'   '+str(img2.shape), fontsize=15)
    ax2.minorticks_on()
    ax2.grid(grid2, which='both')
    # plt.subplots_adjust(left=0.2, right=0.23, top=0.9, bottom=0.)
    plt.show()
   
    
def pixel_histogram(img, RoI_top, RoI_bot, RoI_left = 0, RoI_right = 0, debug = False):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    if RoI_right == 0: 
        RoI_right = img.shape[1] 
    region = img[RoI_top:RoI_bot, RoI_left:RoI_right]
            
    # Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    hg = np.sum(region, axis=0)
    
    midpoint    = np.int(hg.shape[0]//2)
    leftx_base  = np.argmax(hg[:midpoint])
    rightx_base = np.argmax(hg[midpoint:]) + midpoint

    if debug:
        print(' Image shape: ', img.shape, 'Histogram region shape: ', region.shape, 'histogram shape:', hg.shape)
        print(' Region : ', region.shape, RoI_top, RoI_bot)
        print(' Left base: ', leftx_base, ' Right base:', rightx_base)
        display_one(img = region, size = 10)
    
    return hg, region, leftx_base,rightx_base


# def pixel_histogram(img):
    # #Grab only the bottom half of the image
    # # Lane lines are likely to be mostly vertical nearest to the car
    # bottom_half = img[img.shape[0]//2:,:]
    
    # #TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # #i.e. the highest areas of vertical lines should be larger values
    # hg = np.sum(bottom_half, axis=0)
    # print(' Image shape: ', img.shape, 'bottom_half shape: ', bottom_half.shape, 'histogram shape:', hg.shape)
    # return hg

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def displayGuidelines(img, draw = 'both', debug = False):
    """
    helper function to draw marker for camera position and guidelines to assist in 
    marking RoI region
    """
    assert draw in ['x','y','both'], " Invalid draw parm, must be one of 'x', 'y', or 'both' "
    
    height, width = img.shape[:2]
    midline_x = width //2 
    midline_y = height//2
    if draw in ['y', 'both']:
        cv2.drawMarker(img, (midline_x, height-3),(255,255,0), cv2.MARKER_DIAMOND, markerSize=20, thickness=10)
        cv2.line(img, (midline_x, 0), (midline_x, height), (255,255,0), thickness=2)
    if draw in ['x', 'both']:
        cv2.line(img, (0, midline_y), (width , midline_y), (255,255,0), thickness=2)
    
    return img
    
def displayRoILines(img, vertices, thickness = 2, color = 'red', debug = False):
    """
    helper function to draw lines designating the ROI region
    """
    colorRGB = colors.to_rgba_array(color)[0]* 255

    if img.ndim >= 3:
        outputImg = np.copy(img)
    else:
        outputImg = np.dstack((img,img,img))
    # print(' draw_roi():  input : ', img.shape, ' output : ', outputImg.shape)
    
    cv2.line(outputImg, vertices[0], vertices[1], colorRGB, thickness = thickness)
    cv2.line(outputImg, vertices[1], vertices[2], colorRGB, thickness = thickness)
    cv2.line(outputImg, vertices[2], vertices[3], colorRGB, thickness = thickness)
    cv2.line(outputImg, vertices[3], vertices[0], colorRGB, thickness = thickness)
    return outputImg

def colorLanePixelsV1(input_img, leftx, lefty, rightx, righty, debug = False):
    ## Visualization ##
    # Colors in the left and right lane regions
    output_img = np.copy(input_img)
    output_img[lefty, leftx]   = [255, 0, 0]
    output_img[righty, rightx] = [0, 0, 255]
    
    return output_img
    
    
def colorLanePixels(input_img, LLane, RLane, lcolor = 'lightcoral', rcolor = 'lightblue', debug = False):
    if debug: 
        print(' Call displayLanePixels')
    color_left  = colors.to_rgba(lcolor)
    color_right = colors.to_rgba(rcolor)

    result = np.copy(input_img)
    result[LLane.ally, LLane.allx] = color_left    ## [255, 0, 0]  
    result[RLane.ally, RLane.allx] = color_right   ## [0, 0, 255]   
    return result 



def displayText(img, x,y, text, fontHeight = 30, thickness = 2, color = (255,0,0), debug = False): 

    # fontFace   = cv2.FONT_HERSHEY_SIMPLEX
    # lineType   = cv2.LINE_AA
    # output = np.copy(img)

    fontScale = cv2.getFontScaleFromHeight( cv2.FONT_HERSHEY_SIMPLEX, fontHeight , thickness = thickness)
    cv2.putText(img, text, (x, y) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color , thickness, cv2.LINE_AA)
    

    # retval, baseLine= cv2.getTextSize( text, fontFace, fontScale, thickness = thickness)
    # baseLine+ fontHeight
    # if debug:
        # print('font scale: ', fontScale, ' retval : ', etval, 'Baseline: ', baseLine)
    # cv2.putText(output, text, (x, y),fontFace, fontScale, color , thickness, lineType)
    
    return img

    
def draw_lane(img, top, bot, fit, color, debug = False):

    ploty = np.linspace(top, bot-1, bot - top)
    plotx  = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.array([np.vstack([plotx, ploty]).T])
    cv2.polylines(img, np.int_([pts]), False, color, thickness=2, lineType = cv2.LINE_AA)
    
    return img
    


def displayPolynomial(input_img, LLane, RLane, **kwargs):
    assert type(LLane) == type(RLane), 'Second and Third parms must have matching types'    
    # print('displayPolynomial : ', input_img.shape)

    color  = kwargs.setdefault( 'color', 'yellow') 
    start  = kwargs.setdefault('start', 0)  
    debug  = kwargs.setdefault('debug', False)
    iteration = kwargs.setdefault('iteration', -1)
    
    if isinstance(LLane, classes.line.Line) :
        # print(' Its an Line obect')
        try:
            left_fitx   = LLane.xfitted_history[iteration][start:]
            right_fitx  = RLane.xfitted_history[iteration][start:]
            left_ploty  = LLane.ploty[start:]
            right_ploty = RLane.ploty[start:]
        except:
            print(' displayPolynomial() w/ ITERATION=', iteration, 'DOESNT EXIST - IGNORED ')
            return input_img
    elif isinstance(LLane, np.ndarray):
        # print(' Its an numpy array (left_fitx/right_fitx)', LLane.shape, RLane.shape)
        left_fitx    = LLane[start:]
        right_fitx   = RLane[start:]    
        left_height  = LLane.shape[0]
        right_height = RLane.shape[0] 
        left_ploty   = np.linspace(start,  left_height-1,  left_height-start, dtype = np.int)
        right_ploty  = np.linspace(start, right_height-1, right_height-start, dtype = np.int)
    elif isinstance(LLane, collections.deque):
        # print(' Its an deque collection :', len(LLane), len(RLane))
        try:
            left_fitx    = LLane[iteration][start:]
            right_fitx   = RLane[iteration][start:]    
            left_height  = LLane[iteration].shape[0]
            right_height = RLane[iteration].shape[0] 
            left_ploty   = np.linspace(start,  left_height-1,  left_height-start, dtype = np.int)
            right_ploty  = np.linspace(start, right_height-1, right_height-start, dtype = np.int)
        except:
            print(' displayPolynomial() w/ ITERATION=', iteration, 'DOESNT EXIST - IGNORED ')
            return input_img
    else:
        print(' displayPolynomial(): Invalid input parm data type: ', type(LLane))

    left_fitx   = np.int_(np.round_(np.clip(left_fitx ,0, input_img.shape[1]-1), 0))
    right_fitx  = np.int_(np.round_(np.clip(right_fitx,0, input_img.shape[1]-1), 0))
        
    colorRGBA = colors.to_rgba(color)    
    result = np.copy(input_img)
    result[ left_ploty ,  left_fitx] = colorRGBA
    result[right_ploty , right_fitx] = colorRGBA
    
    return  result 


def displayPolySearchRegion(input_img, LLane, RLane, **kwargs):
    assert type(LLane) == type(RLane), 'Second and Third parms must have matching types'    
    # print('displayPolynomial : ', input_img.shape)

    color     = kwargs.setdefault('color', 'springgreen') 
    start     = kwargs.setdefault('start', 0)  
    debug     = kwargs.setdefault('debug', False)
    margin    = kwargs.setdefault('margin', 100)
    iteration = kwargs.setdefault('iteration', -1)

    if debug:
        print('displayPolySearchRegion() ')
        print('  Search margin : ', margin)
    
    if isinstance(LLane, classes.line.Line) :
        # print(' Its an Line obect')
        try:
            left_fitx   = LLane.best_xfitted[start:]
            right_fitx  = RLane.best_xfitted[start:]
            left_ploty  = LLane.ploty[start:]
            right_ploty = RLane.ploty[start:]
        except:
            print(' displayPolynomial() w/ ITERATION=', iteration, 'DOESNT EXIST - IGNORED ')
            return input_img
    elif isinstance(LLane, np.ndarray):
        print(' Its an numpy array (left_fitx/right_fitx)', LLane.shape, RLane.shape)
        # height = input_img.shape[0]
        # left_ploty = np.linspace(start, height-1, height-start)
        # right_ploty = np.linspace(start, height-1, height-start)
        # try:
            # left_fitx   = (left_fit[0] * ploty**2) + (left_fit[1] * ploty) + left_fit[2]
            # right_fitx  = (right_fit[0]* ploty**2) + (right_fit[1]* ploty) + right_fit[2]
        # except TypeError:
            # #  Avoids an error if `left` and `right_fit` are still none or incorrect
            # print('The function failed to fit a line!')
            # left_fitx  = 1*ploty**2 + 1*ploty
            # right_fitx = 1*ploty**2 + 1*ploty

        left_fitx    = LLane[start:]
        right_fitx   = RLane[start:]    
        left_height  = LLane.shape[0]
        right_height = RLane.shape[0] 
        left_ploty   = np.linspace(start,  left_height-1,  left_height-start, dtype = np.int)
        right_ploty  = np.linspace(start, right_height-1, right_height-start, dtype = np.int)
    else:
        print(' displayPolySearchRegion(): Invalid input parm data type: ', type(LLane))

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(input_img)
       
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, left_ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, left_ploty])))])
    left_line_pts     = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, right_ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, right_ploty])))])
    right_line_pts     = np.hstack((right_line_window1, right_line_window2))

    if debug:
        # print('  display using iteration: ', iteration, ' of xfitted_history')
        print('  left_fitx     : ', left_fitx.shape    , '  right_fitx    : ', right_fitx.shape)
        print('  left_line_pts :' , left_line_pts.shape, '  right_line_pts: ' , right_line_pts.shape)
    
    color = colors.to_rgba('springgreen')    
    # Draw the search region onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), color)
    
    result = cv2.addWeighted(input_img, 1, window_img, 0.6, 0)
    
    return result

    
def displayLaneRegion(input_img, LLane, RLane,  Minv, **kwargs):
    ''' 
    LLane, RLane:  Either Line objects or *_fitx numpy arrays
    
    iteration: item from xfitted_history to use for lane region zoning
               -1 : most recent xfitted current_xfitted (==  xfitted_history[-1])
    '''
    # print(' Kwargs: ', kwargs)
    color  = kwargs.setdefault('color', 'green') 
    beta  = kwargs.setdefault('beta', 0.5) 
    start = kwargs.setdefault('start', 0)  
    end   = kwargs.setdefault('end'  , input_img.shape[0]) 
    debug = kwargs.setdefault('debug', False)
    iteration = kwargs.setdefault('iteration', -1)
    if debug:
        print(' displayLaneRegion()')
        print(' input image shape: ', input_img.shape)
        print('iteration ', iteration,  ' start: ', start, '   end: ', end)
    assert type(LLane) == type(RLane), 'Second and Third parms must have matching types'    

    if isinstance(LLane, classes.line.Line) :
        # print(' Its an Line obect')
        left_ploty  = LLane.ploty[start:end]
        left_fitx   = LLane.xfitted_history[iteration][start:end]
        right_ploty = RLane.ploty[start:end]
        right_fitx  = RLane.xfitted_history[iteration][start:end]
    elif isinstance(LLane, np.ndarray):
        # print(' Its an numpy array (left_fitx/right_fitx)')
        # left_height  = LLane.shape[0]
        # right_height = RLane.shape[0]    
        # left_fitx    = LLane
        # right_fitx   = RLane
        # left_ploty   = (np.linspace(start, left_height-1, left_height, dtype = np.int))
        # right_ploty  = (np.linspace(start, right_height-1, right_height, dtype = np.int))
        left_fitx    = LLane[start:end]
        right_fitx   = RLane[start:end]
        left_ploty   = np.linspace(start, end-1, end - start, dtype = np.int)
        right_ploty  = np.linspace(start, end-1, end - start, dtype = np.int)
    else:
        print(' displayLaneRegion(): Invalid input parm data type: ', type(LLane))
        sys.exit(8)
    
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(input_img).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, left_ploty]))]).astype(np.int)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_ploty])))]).astype(np.int)
    pts       = np.hstack((pts_left, pts_right))
    region_color  = colors.to_rgba_array(color)[0] * 255

    if debug:
        print(' left_ploty : ', left_ploty.shape , ' left_fitx : ', left_fitx.shape)
        print(' right_ploty: ', right_ploty.shape, ' right_fitx: ', right_fitx.shape)
        print(' pts_left: {}  pts_right: {}   pts: {}'.format(pts_left.shape, pts_right.shape, pts.shape))    
        print(' poly region color: ', region_color)
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, ([pts]), region_color)
    
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

def displayFittingInfo(LeftLane, RightLane):
    np_format = {}
    np_format['float'] = lambda x: "%12.6f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =100, formatter = np_format)


    print()
    print('Proposed Polynomial:')
    print('left      : {}    right     : {} '.format(LeftLane.current_fit, RightLane.current_fit))

    if len(LeftLane.fit_history) > 0:
        print()
        print('Current Best Fit Polynomial:')
        print('left      : {}    right     : {} '.format(LeftLane.best_fit, RightLane.best_fit))

        print()
        print('Previous Fitted Polynomials:')
        for ls, rs in zip(reversed(LeftLane.fit_history), reversed(RightLane.fit_history)):
            print('left      : {}    right     : {} '.format(ls, rs))
        print()
        print('Diff beteween current and last accepted polynomial ')
        print('Difference: {}    right     : {} '.format( LeftLane.fit_history[-1]-LeftLane.current_fit, RightLane.fit_history[-1]-RightLane.current_fit)    )

        print()
        print('Diff beteween current and best_fit polynomial ')
        print('Difference: {}    right     : {} '.format( LeftLane.best_fit-LeftLane.current_fit, RightLane.best_fit-RightLane.current_fit)    )

    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =100, formatter = np_format)

    print()
    print('Slope @ Y=: ', LeftLane.y_checkpoints)
    ls = LeftLane.current_slope
    rs = RightLane.current_slope
    diff = np.round(np.array(rs)- np.array(ls),3)
    avg  = np.round((np.array(rs) + np.array(ls))/2,3)
    print('left      : ',ls , '  Avg:', np.round(np.mean(ls),3))
    print('right     : ',rs , '  Avg:', np.round(np.mean(rs),3))
    print('avg       : ',avg)
    print('diff      : ',diff, '  Max[700:480]: ', diff[0:5].max())


    if len(LeftLane.best_fit_history) > 0:
        print()
        print('BEST Slope: ', LeftLane.y_checkpoints)
        ls = LeftLane.best_slope
        rs = RightLane.best_slope
        diff = np.round(np.array(rs)- np.array(ls),3)
        avg  = np.round((np.array(rs) + np.array(ls))/2,3)
        print('left      : ',ls, '  Avg:', np.round(np.mean(ls),3))
        print('right     : ',rs, '  Avg:', np.round(np.mean(rs),3))
        print('avg       : ',avg)
        print('diff      : ',diff, '  Max[700:480]: ', diff[0:5].max())

    print()
    print('Slope history LeftLane : ', ['{:8.3f}'.format(i) for i in  LeftLane.slope])
    print('Slope history RightLane: ', ['{:8.3f}'.format(i) for i in RightLane.slope])
    print('Slope Diff History     : ', ['{:8.3f}'.format(i-j) for i,j in zip(RightLane.slope, LeftLane.slope)])
    
    print('\n')
    print('Radius @ Y: ', LeftLane.y_checkpoints)
    ls = LeftLane.current_curvature
    rs = RightLane.current_curvature
    diff = np.round(np.array(rs)- np.array(ls),3)
    avg  = np.round((np.array(rs) + np.array(ls))/2,3)
    print('left      : ',ls, '  Avg:', np.round(np.mean(ls),3))
    print('right     : ',rs, '  Avg:', np.round(np.mean(rs),3))
    print('avg       : ',avg)
    print('diff      : ',diff, '  Max[700:480]: ', diff[0:5].max())

    if len(LeftLane.best_fit_history) > 0:
        print()
        print('BEST Radiu: ', LeftLane.y_checkpoints)
        ls = LeftLane.best_curvature
        rs = RightLane.best_curvature
        diff = np.round(np.array(rs)- np.array(ls),3)
        avg  = np.round((np.array(rs) + np.array(ls))/2,3)
        print('left      : ', ls, '  Avg:', np.round(np.mean(ls),3))
        print('right     : ', rs, '  Avg:', np.round(np.mean(rs),3))
        print('avg       : ', avg)
        print('diff      : ', diff, '  Max[700:480]: ', diff[0:5].max())
    
    print()
    print('Radius History LeftLane : ', ['{:8.3f}'.format(i) for i in  LeftLane.radius])
    print('Radius History RightLane: ', ['{:8.3f}'.format(i) for i in RightLane.radius])
    print('Radius Diff History (m) : ', ['{:8.3f}'.format(i-j) for i,j in zip(RightLane.radius, LeftLane.radius)])
    print('\n')
    print('Line X @ Y: ', LeftLane.y_checkpoints)
    ls = LeftLane.current_linepos
    rs = RightLane.current_linepos
    diff_pxl = np.round( np.array(rs)- np.array(ls),3)
    diff_mtr = np.round((np.array(rs)- np.array(ls))*LeftLane.MX,3)
    print('left      : ',ls, '  Avg:', np.round(np.mean(ls),3))
    print('right     : ',rs, '  Avg:', np.round(np.mean(rs),3))
    print('diff (pxl): ', diff_pxl)
    print('diff (mtr): ', diff_mtr, '  Max[700:480]: ', diff_mtr[0:5].max())


    if len(LeftLane.best_fit_history) > 0:
        print()
        print('BEST Line : ', LeftLane.y_checkpoints)
        ls = LeftLane.best_linepos
        rs = RightLane.best_linepos
        diff_pxl = np.round( np.array(rs)- np.array(ls),3)
        diff_mtr = np.round((np.array(rs)- np.array(ls))*LeftLane.MX,3)
        print('left      : ',ls , ' Avg:', np.round(np.mean(ls),3))
        print('right     : ',rs , ' Avg:', np.round(np.mean(rs),3))
        print('diff (pxl): ', diff_pxl)
        print('diff (mtr): ', diff_mtr, '  Max[700:480]: ', diff_mtr[0:5].max())
    
    print()
    print('Linebase History LeftLane  (m): ', ['{:8.3f}'.format(i) for i in  LeftLane.line_base_meters])
    print('Linebase History RightLane (m): ', ['{:8.3f}'.format(i) for i in RightLane.line_base_meters])
    print('Line width History (m)        : ', ['{:8.3f}'.format(i-j) for i,j in zip(RightLane.line_base_meters, LeftLane.line_base_meters)])
   
def calcOffCenterV1(y_eval, LLane, RLane, center_x, units = 'm', debug = False):
    '''
    Calculate position of vehicle relative to center of lane
    y_eval:                 y-value where we want radius of curvature
    left_fitx, right_fitx:  x_values at y_eval
    center_x:               center of image, represents center of vehicle
    units   :               units to calculate off_center distance in 
                            pixels 'p'  or meters 'm'
    '''
    mid_point  = LLane.xfitted_history[-1][y_eval] + (RLane.xfitted_history[-1][y_eval] - LLane.xfitted_history[-1][y_eval])//2
    off_center_pxls = mid_point - center_x 
    off_center_mtrs = off_center_pxls * (3.7/700)
    if debug: 
        print(' llane: ', len(LLane.xfitted_history), LLane.xfitted_history[-1].shape, 'Rlane:', len(RLane.xfitted_history), RLane.xfitted_history[-1].shape)
        print('Y: {:4.0f}  Left lane: {:8.3f}  right_lane: {:8.3f}  midpt: {:8.3f}  off_center: {:8.3f} , off_center(mtrs): {:8.3f} '.format(
                y_eval   , LLane.xfitted_history[-1][y_eval], RLane.xfitted_history[-1][y_eval],
                mid_point, off_center_pxls             , off_center_mtrs))
    
    oc = off_center_mtrs if units == 'm' else off_center_pxls
    
    if off_center_pxls != 0 :
        output = str(abs(round(oc,3)))+(' m ' if units == 'm' else ' pxls ')  +('left' if oc > 0 else 'right')+' of lane center'
    else:
        output = 'On lane center'
    return output ## round(off_center_mtrs,3), round(off_center,3)
    
def curvatureMsg(LLane, RLane, units = 'm', iteration = -1,  debug = False):    
    str_units = ' m ' if units == 'm' else ' pxls '
    msg = "Curvature  L: "+str(int(LLane.radius[iteration]))+ str_units+"  R: "+str(int(RLane.radius[iteration]))+str_units    
    return msg
    
def offCenterMsg(LLane, RLane, center_x, units = 'm',  iteration = -1, debug = False):
    '''
    Calculate position of vehicle relative to center of lane
    left_fitx, right_fitx:  x_values at y_eval
    center_x:               center of image, represents center of vehicle
    units   :               units to calculate off_center distance in 
                            pixels 'p'  or meters 'm'
    '''
    mid_point_meters  = LLane.line_base_meters[iteration] + (RLane.line_base_meters[iteration] - LLane.line_base_meters[iteration]) / 2
    mid_point_pixels  = LLane.line_base_pixels[iteration] + (RLane.line_base_pixels[iteration] - LLane.line_base_pixels[iteration]) / 2
    off_center_meters = mid_point_meters  - (center_x * LLane.MX) 
    off_center_pixels = mid_point_pixels  - center_x 
    
    if debug: 
        print()
        print(' offCenterMsg():')
        print(' Meters Y: {:4.0f}  Left lane: {:8.3f}  right_lane: {:8.3f}  midpt: {:8.3f}  off_center: {:8.3f} '.format(
               700, LLane.line_base_meters[iteration], RLane.line_base_meters[iteration], mid_point_meters, off_center_meters))
        print(' Pixels Y: {:4.0f}  Left lane: {:8.3f}  right_lane: {:8.3f}  midpt: {:8.3f}  off_center: {:8.3f} '.format(
               700, LLane.line_base_pixels[iteration], RLane.line_base_pixels[iteration], mid_point_pixels, off_center_pixels))
    
    oc = off_center_meters if units == 'm' else off_center_pixels
    
    if off_center_meters != 0 :
        output = str(abs(round(oc,3)))+(' m ' if units == 'm' else ' pxls ')  +('left' if oc > 0 else 'right')+' of lane center'
    else:
        output = 'On lane center'
        
    return output ## round(off_center_mtrs,3), round(off_center,3)
    

def measure_curvature(y_eval, left_fit, right_fit, units = 'm', MX_denom = 700, MY_denom = 720, debug = False):
    '''
    Calculates the curvature of polynomial functions in pixels.
    y_eval:               y-value where we want radius of curvature
    left_fit, right_fit:  polynomial parameters
    '''
    assert units in ['m', 'p'], "Invalid units parameter, must be 'm' for meters or 'p' for pixels"
    
    left_curve = radius(y_eval, left_fit , units = units)  ## Implement the calculation of the left line here
    right_curve= radius(y_eval, right_fit, units = units)  ## Implement the calculation of the right line here
    avg_curve  = round((left_curve + right_curve)/2,3)
    str_units  = " m" if units == 'm' else " pxls"
    message = "Curvature  L: "+str(int(left_curve))+str_units+"  R: "+str(int(right_curve))+str_units
    
    if debug:
        print("  {:8s}  {:8s}   {:8s}   {:8s} ".format(" y eval" ,"avg pxl", "left_pxl" , "right_pxl"))
        print(" {:8.0f}   {:8.2f}   {:8.2f}   {:8.2f} ".format( y_eval, avg_curve, left_curve, right_curve))
        print(message)
    return  message, avg_curve, left_curve, right_curve


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

def region_of_interest(img, vertices, debug = False):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
     
def unwarpImage(img, nx, ny, cam, dst, debug = False):
    # Pass in your image into this function
    # Write code to do the following steps
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
    
def perspectiveTransform(img, source, dest, debug = False):
    if debug: 
        print('src: ', type(source), source.shape, ' - ',  source)
        print('dst: ', type(dest), dest.shape, ' - ',  dest)

    M = cv2.getPerspectiveTransform(source, dest)
    Minv = cv2.getPerspectiveTransform(dest, source)    

    if debug: 
        print(' M: ', M.shape, ' Minv: ', Minv)
        
    return cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR), M, Minv


def fit_polynomial(leftx, lefty, rightx, righty, debug = False):
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


def find_lane_pixels(binary_warped, LLane, RLane, histRange = None,
                     nwindows = NWINDOWS, 
                     window_margin = WINDOW_SEARCH_MARGIN, 
                     minpix   = MINPIX,
                     maxpix   = MAXPIX, 
                     debug    = False):

    LLane.set_height(binary_warped.shape[0])
    RLane.set_height(binary_warped.shape[0])

    if histRange is None:
        histLeft = 0
        histRight = binary_warped.shape[1]
    else:
        histLeft, histRight = int(histRange[0]), int(histRange[1])
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped, np.ones_like(binary_warped)*255))
    
    # if debug: 
        # print(' binary warped shape: ', binary_warped.shape)
        # print(' out_img shape: ', out_img.shape)
        # display_one(out_img, grayscale = False, title = 'out_img')
        # display_one(binary_warped, title='binary_warped')

    # Take a histogram of the bottom half of the image
    
    histogram = np.sum(binary_warped[2*binary_warped.shape[0]//3:, histLeft:histRight], axis=0)
    if debug:
        print(' histogram shape before padding: ' , histogram.shape)
    
    histogram = np.pad(histogram, (histLeft, binary_warped.shape[1]-histRight))
    if debug:
        print(' histogram shape after padding : ' , histogram.shape) 
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    
    midpoint    = np.int(histogram.shape[0]//2)
    leftx_base  = np.argmax(histogram[:midpoint]) 
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint   

    
    LLane.set_MX(LLane.MX_nom, rightx_base - leftx_base, debug = False)
    RLane.set_MX(RLane.MX_nom, rightx_base - leftx_base, debug = False)
    
    if debug:
        print(' Run find_lane_pixels()  - histRange:', histRange)
        print(' Midpoint:  {} '.format(midpoint))
        print(' Histogram left side max: {}  right side max: {}'.format(np.argmax(histogram[:midpoint]),np.argmax(histogram[midpoint:])))
        print(' Histogram left side max: {}  right side max: {}'.format(leftx_base, rightx_base))
    
        print(' MX                : {}  {} '.format(LLane.MX      , RLane.MX))
        print(' MX_nominator   is : {}  {} '.format(LLane.MX_nom  , RLane.MX_nom))
        print(' MX_denominator is : {}  {} '.format(LLane.MX_denom, RLane.MX_denom))
        
        print(' MY                : {}  {} '.format(LLane.MY      , RLane.MY))
        print(' MY_nominator   is : {}  {} '.format(LLane.MY_nom  , RLane.MY_nom))
        print(' MY_denominator is : {}  {} '.format(LLane.MY_denom, RLane.MY_denom))
    
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
        maxpix = (window_height * window_margin )
    
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
  
    # Current positions to be updated later for each window in nwindows
    leftx_current  = leftx_base
    rightx_current = rightx_base
    
    if debug:
        print(' left x base  : ', leftx_base, '  right x base :', rightx_base )
        print(' window_height: ', window_height)
        print(' nonzero x    : ', nonzerox.shape, nonzerox)
        print(' nonzero y    : ', nonzeroy.shape, nonzeroy)
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
        window_color = colors.to_rgba('green')
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
            # print(' X: ', nonzerox[good_left_inds]) ; print(' Y: ', nonzeroy[good_left_inds])
            print(' right_x_inds : ', right_x_inds[0].shape, ' right_y_indx : ', right_y_inds[0].shape,
                  '  -- good right inds size: ', good_right_inds.shape[0])
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
        LLane.set_linePixels(nonzerox[left_line_inds], nonzeroy[left_line_inds])
        RLane.set_linePixels(nonzerox[right_line_inds], nonzeroy[right_line_inds])
        rc = 1
        
    if debug:
        print()
        print(' leftx : ', LLane.allx.shape, ' lefty : ', LLane.ally.shape)
        print(' rightx : ', RLane.allx.shape, ' righty : ', RLane.ally.shape)

    return rc, out_img, histogram
    

def search_around_poly(binary_warped, LLane, RLane, search_margin = POLY_SEARCH_MARGIN, debug = False):
    '''
    # HYPERPARAMETER
    # search_margin : width of the margin around the previous polynomial to search
    '''
    POLY_SEARCH_MIN_THRESHOLD = 1000
    out_img = np.dstack((binary_warped, binary_warped, binary_warped, np.ones_like(binary_warped)))*255

    # Take a histogram of the bottom half of the image
    histogram   = np.sum(binary_warped[2*binary_warped.shape[0]//3:,:], axis=0)
    midpoint    = np.int(histogram.shape[0]//2)
    leftx_base  = np.argmax(histogram[:midpoint]) 
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint     
    
    if debug:
        print(' Run search_around_poly()')
        print('   Histogram Midpoint:  {} '.format(midpoint))
        print('   Histogram left side max: {}  right side max: {}'.format(np.argmax(histogram[:midpoint]),np.argmax(histogram[midpoint:])))
        print('   Histogram left side max: {}  right side max: {}'.format(leftx_base, rightx_base))
        
        
    # Grab activated pixels
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    # left_fit  = LLane.current_fit
    # right_fit = RLane.current_fit

    left_fit  = LLane.best_fit
    right_fit = RLane.best_fit

    fitted_x_left     = (left_fit [0]*nonzeroy**2) + ( left_fit[1]*nonzeroy) + left_fit[2]
    fitted_x_right    = (right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + right_fit[2]
        
    left_line_inds  = ( (nonzerox > ( fitted_x_left - search_margin )) & (nonzerox < (fitted_x_left + search_margin)) ).nonzero()
    right_line_inds = ( (nonzerox > (fitted_x_right - search_margin)) & (nonzerox < (fitted_x_right + search_margin)) ).nonzero()
    
    leftPixelCount  =  left_line_inds[0].shape[0]
    rightPixelCount = right_line_inds[0].shape[0]
    if debug:
        print(' Search_around_poly() ')
        print('   fitted_x_left  : ', fitted_x_left.shape     , '  fitted_x_right : ', fitted_x_right.shape)
        print('   left_lane_inds : ', leftPixelCount  )
        print('   right_lane_inds: ', rightPixelCount )
    
    
    
    if (leftPixelCount < POLY_SEARCH_MIN_THRESHOLD) or (rightPixelCount < POLY_SEARCH_MIN_THRESHOLD)  :
        print(' Insufficient lane pixels detected: LeftLane: ' , leftPixelCount, ' right lane: ', rightPixelCount)
        rc = 0
    else:
        rc = 1
        # Extract left and right line pixel positions
        LLane.set_linePixels(nonzerox [left_line_inds], nonzeroy[left_line_inds])
        RLane.set_linePixels(nonzerox[right_line_inds], nonzeroy[right_line_inds])
    
    return rc,out_img, histogram
  
    # return result


    
"""    
def displayPolySearchRegion2(input_img, left_fit, right_fit, margin = 100,  debug = False):
    color = colors.to_rgba('springgreen')

    # Generate y values for plotting
    height = input_img.shape[0]
    ploty = np.linspace(0, height-1, height)

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(input_img)
    
    try:
        left_fitx   = (left_fit[0] * ploty**2) + (left_fit[1] * ploty) + left_fit[2]
        right_fitx  = (right_fit[0]* ploty**2) + (right_fit[1]* ploty) + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx  = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts     = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts     = np.hstack((right_line_window1, right_line_window2))

    if debug:
        print('displayPolySearchRegion2() ')
        print(' left fit parms : ', left_fit)
        print(' right_fit_parms: ', right_fit)
        print(left_line_pts.shape, right_line_pts.shape)
        
    # Draw the search region onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), color)
    
    result = cv2.addWeighted(input_img, 1, window_img, 0.6, 0)
    
    return result   
  
  
def displayPolynomial2(input_img, left_fitx, right_fitx, color = 'red', debug = False):

    print('displayPolynomial2 : ', input_img.shape)    
    # Generate y values for plotting

    left_height  = left_fitx.shape[0]
    right_height = right_fitx.shape[0]
    
    left_ploty   = (np.linspace(0, left_height-1, left_height, dtype = np.int))
    right_ploty  = (np.linspace(0, right_height-1, right_height, dtype = np.int))
    
    colorRGBA = colors.to_rgba(color)
    result = np.copy(input_img)
    result[left_ploty, np.int_(np.round_(left_fitx ,0))] = colorRGBA
    result[left_ploty, np.int_(np.round_(right_fitx,0))] = colorRGBA
    
    return  result 
 
 
def calc_off_center(y_eval, left_fitx, right_fitx, center_x, units = 'm', debug = False):
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

    
    
def displayLaneRegion1(input_img, LLane, RLane,  Minv, **kwargs):
    '''
    iteration: item from xfitted_history to use for lane region zoning
               -1 : most recent xfitted current_xfitted (==  xfitted_history[-1])
    '''
    beta  = kwargs.setdefault( 'beta', 0.5) 
    start = kwargs.setdefault('start', 0)  
    debug = kwargs.setdefault('debug', False)
    iteration = kwargs.setdefault('iteration', -1)
   
    left_ploty  = LLane.ploty[start:]
    left_fitx   = LLane.xfitted_history[iteration][start:]
    right_ploty = RLane.ploty[start:]
    right_fitx  = RLane.xfitted_history[iteration][start:]

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

    
def displayLaneRegion2(input_img, input1, input2,  Minv, **kwargs ):
    ''' 
    
    '''
    beta  = kwargs.setdefault( 'beta', 0.5) 
    start = kwargs.setdefault('start', 0)  
    debug = kwargs.setdefault('debug', False)
    
    left_height  = left_fitx.shape[0]
    right_height = right_fitx.shape[0]    
    left_ploty   = (np.linspace(start, left_height-1, left_height, dtype = np.int))
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



def displayPolynomial(input_img, left_fit, right_fit, color = [255,255,0], debug = False):
    # Generate y values for plotting

    height    = input_img.shape[0]
    ploty     = (np.linspace(0, height-1, height, dtype = np.int))
    print(ploty)
    left_fit  = LLane.current_xfitted
    right_fit = RLane.best_fit
    try:
        left_fitx  = np.int_(np.round_(left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] ,0))
        right_fitx = np.int_(np.round_(right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] ,0))
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx  = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
    result = np.copy(input_img)
    result[ploty, left_fitx ] = color
    result[ploty, right_fitx] = color
    
    return  result 


    
    
def find_lanes_and_fit_polynomial(binary_warped, debug = False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    left_fit, right_fit          = fit_polynomial(leftx, lefty, rightx, righty, out_img)
    ploty, left_fitx, right_fitx = plot_polynomial(binary_warped.shape[0], left_fit, right_fit)

    output_img  = display_lane_pixels(out_img, leftx, lefty, rightx, righty)
    
    return ploty, left_fitx, right_fitx, output_img
"""

"""    
def colorLanePixels(input_img, LeftLane, RightLane, debug = False):
    ## Visualization ##
    # Colors in the left and right lane regions
    if debug: 
        print(' Call displayLanePixels')
    result = np.copy(input_img)
    result[LeftLane.ally, LeftLane.allx]   = [255, 0, 0]
    result[RightLane.ally, RightLane.allx] = [0, 0, 255]    
    return result
"""
    
"""
def radius(y_eval, fit_coeffs, units, MX_denom = 700, MY_denom = 720, debug = False):
    MY = 30/MY_denom # meters per pixel in y dimension
    MX= 3.7/MX_denom # meters per pixel in x dimension
    A,B,_ = fit_coeffs   
    if units == 'm':
        A = (A * MX)/ (MY**2)
        B = (B * MX/MY)
    
    return  ((1 + ((2*A*(y_eval*MY))+B)**2)** 1.5)/np.absolute(2*A)    
"""
    