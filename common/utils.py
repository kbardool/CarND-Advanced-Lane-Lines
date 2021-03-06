import sys,os, pprint
if '..' not in sys.path:
    print("utils.py: appending '..' to sys.path")
    sys.path.append('..')

import numpy as np
import cv2
import glob
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from   matplotlib import colors
import pickle
import pprint 
import collections

pp = pprint.PrettyPrinter(indent=2, width=100)
print(' Loading utils.py - cwd:', os.getcwd())
  
# from classes.line import Line ## import classes.line as line
import classes.line
RED   = 0
GREEN = 1
BLUE  = 2


NWINDOWS = 9
MINPIX   = 90
MAXPIX   = 8000
WINDOW_SEARCH_MARGIN  = 100
POLY_SEARCH_MARGIN    = 100
PIXEL_THRESHOLD       = 500
PIXEL_RATIO_THRESHOLD = 30
LANE_RATIO_THRESHOLD  = 10


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    fig , ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig,ax
    
def display_one(img, title = 'Input image ', cmap = 'jet' , grayscale = False,
    ax = None, size = (12,9) , winttl= '', grid = None, clim = None,  cbar = False):
    
    if ax is None:
        fig , ax = plt.subplots(1, 1, figsize=size)
         
    fig.canvas.set_window_title(winttl+' : '+ title)        
    if img.ndim == 1:    ## histrograms
        x = np.arange(0,img.shape[0],1) 
        cs = ax.plot(img)
    else:

        cs = plt.imshow(img, cmap = cmap)
    
    # if grayscale and img.ndim == 3:
        # imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # ax.imshow(imgGray, cmap='gray')
    # else:
        # ax.imshow(img, cmap = cmap)

    if grid is not None:
        ax.minorticks_on()
        ax.grid(True, which=grid)    
    
    if clim is not None:
        plt.clim(clim)    
    
    if cbar:
        fig.colorbar(cs, shrink=0.5, aspect=20, fraction=0.05)

    ax.set_title(title+'  '+str(img.shape))
    
    plt.show()


def display_one_cbar(image, title = 'Image', cmap = 'jet'):
    return display_one(image, title, size=(15,10))

# def display_arctan(image, title = 'Image', cmap = 'jet' , ax = None, clim = None,  size = (12,9)):
    
#     if ax is None:
#         fig , ax = plt.subplots(1, 1, figsize=size)
    
#     cs = plt.imshow(image, cmap=cmap)
#     fig.colorbar(cs, shrink=0.5, aspect=20, fraction=0.05)
#     if clim is not None:
#         plt.clim(clim)
#     plt.title(title)
#     plt.show()    
#     return

def display_two(img1, img2, title1 = None , title2 =None, 
                size = (24,9), winttl = ' ', grid1 = None, grid2 = None, fontsize = 15):
    assert type(size) == tuple, 'size must be a tuple : (width, height) '
    if title1 is None:
        title1 = 'Original Image:'+'   '+str(img1.shape)
        if title2 is None:
            title2 = 'Result Image '+'   '+str(img2.shape)
    else:
        if title2 is None:
            title2 = title1

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
    f.canvas.set_window_title(winttl+' : '+ title1+' - '+title2)
    f.tight_layout()
    
    cmap = 'gray' if img1.ndim == 2 else 'jet'
    if img1.ndim == 1:
        ax1.plot(img1)
    else:
        ax1.imshow(img1, cmap = cmap)
    ax1.set_title(title1, fontsize=fontsize)
    ax1.minorticks_on()
    
    if grid1 is not None:
        ax1.grid(True, which=grid1)    
        # ax1.grid(True,which = 'both', axis='both', color='r', linestyle = '-', linewidth = 1)

    cmap = 'gray' if img2.ndim == 2 else 'jet'
    if img2.ndim == 1:
        ax2.plot(img2)
    else:
        ax2.imshow(img2, cmap = cmap)
    ax2.set_title(title2, fontsize=fontsize)
    ax2.minorticks_on()
    
    if grid2 is not None:
        ax2.grid(True, which=grid2)
        # plt.subplots_adjust(left=0.2, right=0.23, top=0.9, bottom=0.)
    
    plt.show()
   

def display_three(*img, **kwargs): 
    assert len(img) == 3 , 'requires 3 image inputs'
    title = [] 
    grid  = []
    for i in range(len(img)):
        assert isinstance(img[i], np.ndarray), ' img'+str(i+1)+' must be numpy.ndarray type' 
        title.append( kwargs.get('title'+str(i+1), 'image '+str(i)+'  '+str(img[i].shape) ))
        grid.append( kwargs.get('grid'+str(i+1) , None))
    size     = kwargs.get('size' , (24,9))
    winttl   = kwargs.get('winttl' , '')
    assert type(size) == tuple, 'size must be a tuple : (width, height) '
    
    f, ax = plt.subplots(1, 3, figsize=size)
    f.canvas.set_window_title(winttl)
    f.tight_layout()
    
    for i in range(3):
        cmap = 'gray' if img[i].ndim == 2 else 'jet'
        if img[i].ndim == 1:
            ax[i].plot(img[i])
        else:
            ax[i].imshow(img[i], cmap = cmap)
        ax[i].set_title(title[i], fontsize=15)
        ax[i].minorticks_on()
    
        if grid[i]:
            ax[i].grid(True, which=grid[i])    
            # ax1.grid(True, which = 'both', axis='both', color='r', linestyle = '-', linewidth = 1)
    plt.show()


def display_four(*img, **kwargs): 
    assert len(img) == 4 , 'requires 4 image inputs'
    title = [] 
    grid  = []
    for i in range(len(img)):
        assert isinstance(img[i], np.ndarray), ' img'+str(i+1)+' must be numpy.ndarray type'         
        title.append( kwargs.get('title'+str(i+1), 'image '+str(i+1)+'  '+str(img[i].shape) ))
        grid.append( kwargs.get('grid'+str(i+1) , None))
    size     = kwargs.get('size' , (24,9))
    winttl   = kwargs.get('winttl' , '')
    assert type(size) == tuple, 'size must be a tuple : (width, height) '
    
    f, ax = plt.subplots(1, 4, figsize=size)
    f.canvas.set_window_title(winttl)
    f.tight_layout()
    
    for i in range(len(img)):
        cmap = 'gray' if img[i].ndim == 2 else 'jet'
        if img[i].ndim == 1:
            ax[i].plot(img[i])
        else:
            ax[i].imshow(img[i], cmap = cmap)
        ax[i].set_title(title[i], fontsize=15)
        ax[i].minorticks_on()
    
        if grid[i]:
            ax[i].grid(True, which=grid[i])    
            # ax1.grid(True, which = 'both', axis='both', color='r', linestyle = '-', linewidth = 1)
    plt.show()


def display_multi(*img, **kwargs): 
    title = [] 
    grid  = []
    for i in range(len(img)):
        assert isinstance(img[i], np.ndarray), ' img'+str(i+1)+' must be numpy.ndarray type'         
        title.append( kwargs.get('title'+str(i+1), 'image '+str(i+1)+'  '+str(img[i].shape) ))
        grid.append( kwargs.get('grid'+str(i+1) , None))
    size     = kwargs.get('size' , (24,9))
    winttl   = kwargs.get('winttl' , '')
    assert type(size) == tuple, 'size must be a tuple : (width, height) '
    
    f, ax = plt.subplots(1, len(img), figsize=size)
    f.canvas.set_window_title(winttl)
    f.tight_layout()
    
    for i in range(len(img)):
        cmap = 'gray' if img[i].ndim == 2 else 'jet'
        if img[i].ndim == 1:
            ax[i].plot(img[i])
        else:
            ax[i].imshow(img[i], cmap = cmap)
        ax[i].set_title(title[i], fontsize=15)
        ax[i].minorticks_on()
    
        if grid[i]:
            ax[i].grid(True, which=grid[i])    
            # ax1.grid(True, which = 'both', axis='both', color='r', linestyle = '-', linewidth = 1)
    plt.show()

def draw_lane(img, top, bot, fit, color, debug = False):
    ploty = np.linspace(top, bot-1, bot - top)
    plotx  = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.array([np.vstack([plotx, ploty]).T])
    cv2.polylines(img, np.int_([pts]), False, color, thickness=2, lineType = cv2.LINE_AA)
    return img
    
def displayText(img, x,y, text, fontHeight = 30, thickness = 2, color = (255,0,0), debug = False): 
    fontScale = cv2.getFontScaleFromHeight( cv2.FONT_HERSHEY_SIMPLEX, fontHeight , thickness = thickness)
    cv2.putText(img, text, (x, y) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color , thickness, cv2.LINE_AA)
    if debug:
        print(' displayText()')
        print(f' x: {x}  y: {y:10d}  text:{text}   fontscale: {fontScale} ')
    return img

def displayGuidelines(img, midline_y = None, midline_x = None, draw = 'both', debug = False):
    '''
    helper function to draw marker for camera position and guidelines to assist in 
    marking RoI region
    '''
    assert draw in ['x','y','both'], " Invalid draw parm, must be one of 'x', 'y', or 'both' "
    
    height, width = img.shape[:2]
    if midline_x is None:
        midline_x = width //2 
    if midline_y is None:
        midline_y = height//2
    if draw in ['y', 'both']:
        cv2.drawMarker(img, (midline_x, height-3),(255,255,0), cv2.MARKER_DIAMOND, markerSize=20, thickness=10)
        cv2.line(img, (midline_x, 0), (midline_x, height), (255,255,0), thickness=2)
    if draw in ['x', 'both']:
        cv2.line(img, (0, midline_y), (width , midline_y), (255,255,0), thickness=2)
    
    return img
    
def displayRoILines(img, vertices, thickness = 2, color = 'red', debug = False):
    '''
    helper function to draw lines designating the ROI region
    vertices: list of (x,y)  RoI points: [ top left, top right, bottom right, bottom left] 
    '''
    colorRGB = colors.to_rgba_array(color)[0]* 255
    
    if img.ndim >= 3:
        outputImg = np.copy(img)
    else:
        outputImg = np.dstack((img,img,img))
    
    cv2.line(outputImg, vertices[0], vertices[1], colorRGB, thickness = thickness)
    cv2.line(outputImg, vertices[1], vertices[2], colorRGB, thickness = thickness)
    cv2.line(outputImg, vertices[2], vertices[3], colorRGB, thickness = thickness)
    cv2.line(outputImg, vertices[3], vertices[0], colorRGB, thickness = thickness)

    x_upper = (vertices[1][0] - vertices[0][0])//2 + vertices[0][0] 
    x_lower = (vertices[2][0] - vertices[3][0])//2 + vertices[3][0] 
    y_upper = vertices[0][1]
    y_lower = vertices[3][1]
    
    cv2.line(outputImg, (x_upper, y_upper), (x_lower, y_lower), (255,255,0), thickness=2)

    return outputImg

    
def colorLanePixels(input_img, LLane, RLane, lcolor = 'red', rcolor = 'blue', debug = False):
    '''
    Color left/right lane pixels detected in find_lane_pixels() for visualization
    '''
    if debug: 
        print(' Call displayLanePixels')
    if input_img.shape[-1] == 4:
        color_left  = colors.to_rgba_array(lcolor)[0]* 255
        color_right = colors.to_rgba_array(rcolor)[0]* 255
    else:
        color_left  = colors.to_rgb_array(lcolor)[0]* 255
        color_right = colors.to_rgb_array(rcolor)[0]* 255

    result = np.copy(input_img)
    result[LLane.ally, LLane.allx] = color_left    ## [255, 0, 0]  
    result[RLane.ally, RLane.allx] = color_right   ## [0, 0, 255]   
    return result 

  
def curvatureMsg(LLane, RLane, units = 'm', iteration = -1,  debug = False):    
    assert units in ['m', 'p'], "Invalid units parameter, must be 'm' for meters or 'p' for pixels"
    str_units = ' m ' if units == 'm' else ' pxls '

    # msg = "Curvature  L: "+str(int(LLane.radius[iteration]))+ str_units+"  R: "+str(int(RLane.radius[iteration]))+str_units    
    message = "Curvature  L: "+str(int(LLane.radius_avg))+ str_units+"  R: "+str(int(RLane.radius_avg))+str_units    

    if debug: 
        print()
        print(' curvatureMsg():')
        print(message)
    return message


def displayPolynomial(input_img, leftInput, rightInput, **kwargs):
    assert type(leftInput) == type(rightInput), 'Second and Third parms must have matching types'    
    '''
     display a Line.fitted_* plot line
    '''
    # print(' displayPolynomial(): ', input_img.shape)
    color        = kwargs.get( 'color', 'yellow') 
    start        = kwargs.get('start', 0)  
    end          = kwargs.get('end'  , input_img.shape[0])     
    debug        = kwargs.get('debug', False)
    iteration    = kwargs.get('iteration', -1)
    thickness    = kwargs.get('thickness',  1)
    
    img_height, img_width  = input_img.shape[0:2]
    if isinstance(leftInput, classes.line.Line) :
        # print(' displayPolynomial(): Input is a Line object')
        LLane  = leftInput.fitted_history[iteration][0,start:]
        RLane  = rightInput.fitted_history[iteration][0,start:]
    
    elif isinstance(leftInput, np.ndarray):
        # print(' displayPolynomial(): Input is a numpy array (left_fitx/right_fitx )', leftInput.shape, rightInput.shape)
        LLane  = leftInput
        RLane  = rightInput

    elif isinstance(leftInput, collections.deque):
        # print(' displayPolynomial(): Input is a deque collection :', len(leftInput), len(rightInput))
        try:
            LLane = leftInput[iteration][start:]
            RLane = rightInput[iteration][start:]    
        except:
            # print(' displayPolynomial() w/ ITERATION=', iteration, 'DOESNT EXIST - IGNORED ')
            return input_img
    else:
        print(' displayPolynomial(): Invalid input parm data type: ', type(leftInput), ' R:' ,type(rightInput))

    left_idx  = (end > LLane[1,:]) & (LLane[1,:] >= start) 
    right_idx = (end > RLane[1,:]) & (RLane[1,:] >= start) 
    left_x    = np.int_(LLane[0,left_idx])
    left_y    = np.int_(LLane[1,left_idx])
    right_x   = np.int_(RLane[0,right_idx])
    right_y   = np.int_(RLane[1,right_idx])
    left_x   = np.clip(left_x ,0, input_img.shape[1]-1)
    right_x  = np.clip(right_x,0, input_img.shape[1]-1)
        
    colorRGBA = colors.to_rgba_array(color)[0]*255    
    result = np.copy(input_img)
    for i in range(0,thickness,1):
        plot_left_x  = np.clip(left_x + thickness , 0, img_width - 1)
        plot_right_x = np.clip(right_x + thickness, 0, img_width - 1)
        # print(' plot_left_x : ', plot_left_x.min(), plot_left_x.max(), ' plot_right_x:', plot_right_x.min(), plot_right_x.max())
        result[ left_y ,  plot_left_x ] = colorRGBA
        result[right_y ,  plot_right_x] = colorRGBA
    
    return  result 


def displayPolySearchRegion(input_img, leftInput, rightInput, **kwargs):
    assert type(leftInput) == type(rightInput), 'Second and Third parms must have matching types'    
    # print('displayPolynomial : ', input_img.shape)
    wcolor        = kwargs.get('color', 'springgreen') 
    start         = kwargs.get('start', 0)  
    end           = kwargs.get('end'  , input_img.shape[0]) 
    debug         = kwargs.get('debug', False)
    search_margin = kwargs.get('search_margin', 100)
    iteration     = kwargs.get('iteration', -1)

    if debug:
        print()
        print('DisplayPolySearchRegion() ')
        print('  Search margin : ', search_margin)
    
    if isinstance(leftInput, classes.line.Line) :
        # print(' displayPolySearchRegion(): input is a Line object')
        LLane = leftInput.fitted_best
        RLane = rightInput.fitted_best

    elif isinstance(leftInput, np.ndarray):
        # print(' displayPolySearchRegion(): input is a numpy array (left_fitx/right_fitx)', LLane.shape, RLane.shape)
        LLane = leftInput
        RLane = rightInput
    else:
        print(' displayPolySearchRegion(): Invalid input parm data type: ', type(leftInput), ' R:' ,type(rightInput))

    # left_idx  = (end > LLane[1,:]) & (LLane[1,:] >= start) 
    # right_idx = (end > RLane[1,:]) & (RLane[1,:] >= start) 
    # left_x    = np.int_(LLane[0,left_idx])
    # left_y    = np.int_(LLane[1,left_idx])
    # right_x   = np.int_(RLane[0,right_idx])
    # right_y   = np.int_(RLane[1,right_idx])

    left_x    = LLane[0,:]
    left_y    = LLane[1,:]
    right_x   = RLane[0,:]
    right_y   = RLane[1,:]

       
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    
    left_line_window1 = np.array([np.transpose(np.vstack([left_x - search_margin, left_y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_x + search_margin, left_y])))])
    left_line_pts     = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_x - search_margin, right_y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_x + search_margin, right_y])))])
    right_line_pts     = np.hstack((right_line_window1, right_line_window2))

    if debug:
        # print('  display using iteration: ', iteration, ' of xfitted_history')
        print('  left_fitx     : ', left_x.shape    , '  right_fitx    : ', right_x.shape)
        print('  left_line_pts :' , left_line_pts.shape, '  right_line_pts: ' , right_line_pts.shape)
    
    wcolor = colors.to_rgba_array('springgreen')[0] * 255    
    
    # Create an blank array to draw the search region on 
    # Draw the search region onto the warped blank image
    window_img = np.zeros_like(input_img)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), wcolor)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), wcolor)
    result = cv2.addWeighted(input_img, 1, window_img, 0.6, 0)
    
    return result
    

def displayDetectedRegion(input_img, leftInput, rightInput,  Minv, **kwargs):
    ''' 
    Overlay interlane detected region over undistorted image 

    Parameters
    ----------
    leftInput,     Left/Right lane objects or pixel array of fitted lines
    rightInput

    Minv
    LLane, RLane:  Either Line objects or *_fitx numpy arrays
    
    iteration: item from xfitted_history to use for lane region zoning
               -1 : most recent xfitted current_xfitted (==  xfitted_history[-1])
    '''
    assert type(leftInput) == type(rightInput), 'Second and Third parms must have matching types'    

    # print(' Kwargs: ', kwargs)
    color         = kwargs.get('color', 'green') 
    alpha         = kwargs.get('alpha', 1.0) 
    beta          = kwargs.get('beta' , 0.5) 
    disp_start    = kwargs.get('disp_start',  0)  
    disp_end      = kwargs.get('disp_end'  , input_img.shape[0]) 
    debug         = kwargs.get('debug', False)
    frameTitle    = kwargs.get('frameTitle', '')
    iteration     = kwargs.get('iteration', -1)
    region_color  = colors.to_rgba_array(color)[0] * 255

    if debug:
        print('\ndisplayDetectedRegion()')
        print('-'*25)
        print('    Display region between start: ', disp_start, ' -  end: ', disp_end)
    if isinstance(leftInput, classes.line.Line) :
        # print('    displayDetectedRegion(): input is a Line obect')
        LLane = leftInput.fitted_history[iteration]
        RLane = rightInput.fitted_history[iteration]
    elif isinstance(leftInput, np.ndarray):
        # print('    displayDetectedRegion(): input is a numpy array (left_fitx/right_fitx)')
        LLane = leftInput
        RLane = rightInput
    else:
        print('    displayDetectedRegion(): Invalid input parm data type: ', type(leftInput), ' R:' ,type(rightInput))
        sys.exit(8)

    # Create an image to draw the lines on
    detectionOverlay  = np.zeros_like(input_img)

    pts_left  = np.expand_dims(LLane.T.astype(np.int), 0)
    pts_right = np.expand_dims(np.flipud(RLane.T.astype(np.int)),0)

    dst_pts_left  = cv2.perspectiveTransform(pts_left.astype(np.float), Minv).astype(np.int32)
    dst_pts_right = cv2.perspectiveTransform(pts_right.astype(np.float), Minv).astype(np.int32)

    dyn_src_points_list =     [ (dst_pts_left[0,0,0]  , dst_pts_left[0,0,1]  ),
                                (dst_pts_right[0,-1,0], dst_pts_right[0,-1,1]), 
                                (dst_pts_right[0,0,0] , dst_pts_right[0,0,1] ),
                                (dst_pts_left[0,-1,0] , dst_pts_left[0,-1,1] ),]

    disp_left_idx  = (disp_end >  dst_pts_left[0,:,1]) & ( dst_pts_left[0,:,1] >= disp_start) 
    disp_right_idx = (disp_end > dst_pts_right[0,:,1]) & (dst_pts_right[0,:,1] >= disp_start) 
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    disp_pts_left  = dst_pts_left[:,disp_left_idx,:]
    disp_pts_right = dst_pts_right[:,disp_right_idx,:]
    disp_pts       = np.hstack((disp_pts_left, disp_pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(detectionOverlay, ([disp_pts]), region_color)

    # draw left and right lanes
    cv2.polylines(detectionOverlay, (disp_pts_left[:,10:]) , False, (255,0,0), thickness=10, lineType = cv2.LINE_8)
    cv2.polylines(detectionOverlay, (disp_pts_right[:,:-10]), False, (0,0,255), thickness=10, lineType = cv2.LINE_8)
    
    # Combine the unwarped detected lanes result with the original image
    result = cv2.addWeighted(input_img, alpha , detectionOverlay, beta, 0)

    if debug:
        # print(' iteration  : ', iteration,  ' start: ', start, '   end: ', end)
        # print('    left idxs: ', left_idx.shape, ' right x/y', right_idx.shape)
        print('       pts_*      - fitted points passed       - shape: ', pts_left.shape     , pts_right.shape)
        print('       dst_pts_*  - fitted points transformed  - shape: ', dst_pts_left.shape , dst_pts_right.shape)
        print('       disp_*_idx - end > dst_pts > start idxs - shape: ', disp_left_idx.shape, disp_right_idx.shape)
        print('       disp_pts_* - end > dst_pts > start      - shape: ', disp_pts_left.shape, disp_pts_right.shape)
        print('       disp_pts   - combined dst points        - shape: ', disp_pts.shape)

        print('    before warping: ')
        print('       left : {}  X min: {:8.3f}  X max: {:8.3f}    Y min: {:8.3f}  Y max: {:8.3f}'.format(pts_left.shape,
                      pts_left[0,:,0].min(),  pts_left[0,:,0].max(), pts_left[0,:,1].min(),  pts_left[0,:,1].max()))
        print('       right: {}  X min: {:8.3f}  X max: {:8.3f}    Y min: {:8.3f}  Y max: {:8.3f}'.format(pts_right.shape, 
                      pts_right[0,:,0].min(), pts_right[0,:,0].max(),pts_right[0,:,1].min(), pts_right[0,:,1].max()))
        print()
        print('    after warping: (dst_pts)')
        print('       left : {}  X min: {:8.3f}  X max: {:8.3f}    Y min: {:8.3f}  Y max: {:8.3f} '.format(dst_pts_left.shape,
                      dst_pts_left[0,:,0].min(),  dst_pts_left[0,:,0].max(), dst_pts_left[0,:,1].min(),  dst_pts_left[0,:,1].max()))
        print('       right: {}  X min: {:8.3f}  X max: {:8.3f}    Y min: {:8.3f}  Y max: {:8.3f}'.format(dst_pts_right.shape, 
                      dst_pts_right[0,:,0].min(), dst_pts_right[0,:,0].max(), dst_pts_right[0,:,1].min(), dst_pts_right[0,:,1].max()))
        print()
        print('    after warping and limiting display region: (dst_pts)')
        print('       left : {}  X min: {:8.3f}  X max: {:8.3f}    Y min: {:8.3f}  Y max: {:8.3f} '.format(disp_pts_left.shape,
                      disp_pts_left[0,:,0].min(),  disp_pts_left[0,:,0].max(), disp_pts_left[0,:,1].min(),  disp_pts_left[0,:,1].max()))
        print('       right: {}  X min: {:8.3f}  X max: {:8.3f}    Y min: {:8.3f}  Y max: {:8.3f}'.format(disp_pts_right.shape, 
                      disp_pts_right[0,:,0].min(), disp_pts_right[0,:,0].max(), disp_pts_right[0,:,1].min(), disp_pts_right[0,:,1].max()))

        # print(' Input image: ', input_img.shape, ' newwarp : ', newwarp.shape)
        # print(' TL: ', dst_pts_left[0,0], '  TR:', dst_pts_right[0,-1],  '  BR: ' , dst_pts_right[0,0] , '  BL: ', dst_pts_left[0,-1])
        # display_two(color_warp, newwarp, title1 = 'color_warp: ', title2 = 'newwarp', grid1 = 'major', grid2 = 'major')    
        # display_two(newwarp, unwarped_image, title1 = 'newwarp: ', title2 = 'newwarp2', grid1 = 'major', grid2 = 'major')    
        # display_two(detectionOverlay , result , title1 = 'unwarped '+frameTitle, title2 = 'result '+frameTitle, grid1 = 'major', grid2 = 'major')    
        
    return result, dyn_src_points_list   

    
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


def offCenterMsg(LLane, RLane, center_x, units = 'm',  iteration = -1, debug = False):
    '''
    Calculate position of vehicle relative to center of detected lanes

    Parameters:
    -----------    
    Calculate position of vehicle relative to center of lane
    left_fitx, right_fitx:  x_values at y_eval
    center_x:               center of image, represents center of vehicle
    units   :               units to calculate off_center distance in 
                            pixels 'p'  or meters 'm'
    '''
    mid_point_meters  = LLane.line_base_meters[iteration] + (RLane.line_base_meters[iteration] - LLane.line_base_meters[iteration]) / 2
    off_center_meters = (center_x * LLane.MX) - mid_point_meters   
    
    mid_point_pixels  = LLane.line_base_pixels[iteration] + (RLane.line_base_pixels[iteration] - LLane.line_base_pixels[iteration]) / 2
    off_center_pixels = center_x - mid_point_pixels  
    
    oc = off_center_meters if units == 'm' else off_center_pixels
    
    if off_center_meters != 0 :
        output = str(abs(round(oc,3)))+(' m ' if units == 'm' else ' pxls ')  +('left' if oc < 0 else 'right')+' of lane center'
    else:
        output = 'On lane center'

    if debug: 
        print()
        print(' offCenterMsg():')
        print(' Meters Y: {:4.0f}  Left lane: {:8.3f}  right_lane: {:8.3f}  midpt: {:8.3f}  off_center: {:8.3f} '.format(
               700, LLane.line_base_meters[iteration], RLane.line_base_meters[iteration], mid_point_meters, off_center_meters))
        print(' Pixels Y: {:4.0f}  Left lane: {:8.3f}  right_lane: {:8.3f}  midpt: {:8.3f}  off_center: {:8.3f} '.format(
               700, LLane.line_base_pixels[iteration], RLane.line_base_pixels[iteration], mid_point_pixels, off_center_pixels))
        print(' Off center message: ' , output)
        
    return output ## round(off_center_mtrs,3), round(off_center,3)


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
     

def pixelHistogram(img, RoI_top, RoI_bot, RoI_left = 0, RoI_right = 0, debug = False):
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
        display_one(img = region, size = (10,5))
    
    return hg, region, leftx_base, rightx_base
    

def saveOutputImage( inputFilename, result, sfx = '', outputPath =  './output_images/',  mode = None):
    assert mode is not None  and (0 < mode < 3) , ' mode must be 1 or 2'
    if mode == 1:
        # print(' Mode 1: Threshold Image --> Warp ' )
        sfx1 = '_mode1'   ### Warped AFTER thresholding
    else:
        # print(' Mode 2: Warp Image  --> Thresholding ' )
        sfx1 = '_mode2'   ### Warped BEFORE thresholding

    in_filename_ext = os.path.basename(inputFilename)
    in_filename     = os.path.splitext(in_filename_ext)
    out_filename    = in_filename[0]+'_output'+sfx1+sfx+in_filename[1]
    save_filename   = outputPath+out_filename
    display_one(result, winttl = save_filename, title = in_filename_ext +' - Final Result')
    try:
        rc = mpimg.imsave(save_filename, result)
    except Exception as e:
        print(' image save failed  ')
        print(e)
    else:
        print(' Saved to : ' , save_filename)


def loadCameraCalibration(configFilename):
    with open(configFilename, 'rb') as infile:
        camera = pickle.load(infile)
        print(' Camera calibration file loaded ...')
        
    print()
    print(' Camera Calibration Matrix :')
    print(' ---------------------------')
    print(camera.cameraMatrix)
    return camera


def sliding_window_detection(binary_warped, LLane, RLane, **kwargs):
    histRange         = kwargs.get('histRange'     ,  None)  
    debug             = kwargs.get('debug'         ,  False)  
    debug2            = kwargs.get('debug2'        ,  False)  
    minpix            = kwargs.get('minpix'        ,  MINPIX)  
    maxpix            = kwargs.get('maxpix'        ,  0)  
    nwindows          = kwargs.get('nwindows'      ,  NWINDOWS)  
    search_margin     = kwargs.get('search_margin' ,  WINDOW_SEARCH_MARGIN)
    window_color      = kwargs.get('window_color'  ,  'green')  
    # histDepthRange    = kwargs.get('histDepthRange',  binary_warped.shape[0]) 
    histWidthRange    = kwargs.get('histWidthRange',  binary_warped.shape[1])  
    reset_search_base = kwargs.get('reset_search_base'  ,  True)  

    img_height = binary_warped.shape[0]
    img_width  = binary_warped.shape[1]
    
    Left_lower_margin   = search_margin 
    Left_higher_margin  = search_margin 
    Right_lower_margin  = search_margin  
    Right_higher_margin = search_margin  

    window_height = img_height//nwindows
    
    # Create empty lists to receive left and right lane pixel indices
    left_line_inds  = []
    right_line_inds = []

    if maxpix == 0 :
        maxpix = (window_height * 2 * search_margin)

    ttlImagePixels = img_height * img_width
    laneSearchPixels = (window_height * 2 * search_margin) * nwindows
    ttlSearchPixels = 2 * laneSearchPixels

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped , binary_warped, binary_warped,  np.ones_like(binary_warped)))
    out_img *= 255
            
    LLane.set_height(img_height)
    RLane.set_height(img_height)

    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if debug:
        print()
        print('sliding _window_detection()' )
        print('-'*20) 
        print(' NWindows: {}   window search margi : {}   windows_width: {}   window_height:  {}    minpix: {}    maxpix : {} '.format(
                nwindows, search_margin ,  2* search_margin , window_height,  minpix,  maxpix))
        if len(LLane.x_base) > 1:
            print(' Prev-2   X Base Left      : {}  right: {}'.format(LLane.x_base[-2], RLane.x_base[-2]))
        print(' Prev-1   X Base Left      : {}  right: {}'.format(LLane.x_base[-1], RLane.x_base[-1]))
        print()

    ##------------------------------------------------------------------------------------------------
    # Take a histogram of the bottom half of the image    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the -left and right lines
    # if previous starting points do not exist 
    ##------------------------------------------------------------------------------------------------

    histLeft  = max(img_width//2 - histWidthRange, 0)  
    histRight = min(img_width//2 + histWidthRange, img_width)  
    histTop   = 3*img_height //4  ## min(histDepthRange, img_height)
    histBot   = img_height 
    
    histogram = np.sum(binary_warped[histTop : histBot, histLeft : histRight], axis=0)
    hg_shape_before = histogram.shape[0]
    histogram = np.pad(histogram, (histLeft, binary_warped.shape[1]-histRight))
    midpoint  = np.int(histogram.shape[0]//2)

    ##------------------------------------------------------------------------------------------------
    # If a reset of the lane base points is necessary they are reset based on results ftom the 
    # pixel histogram. Otherwise, they are set to the base points of the previous frame
    ##------------------------------------------------------------------------------------------------
    # if (LLane.x_base == 0) and (RLane.x_base == 0):
    if reset_search_base:
        if debug:
            print(' reset_x_base ', reset_search_base)
        # LLane.x_base.append(np.argmax(histogram[:midpoint])) 
        # RLane.x_base.append(np.argmax(histogram[midpoint:]) + midpoint)
        win_xleft_center  =  np.argmax(histogram[:midpoint]) 
        win_xright_center =  np.argmax(histogram[midpoint:]) + midpoint
    else:
        win_xleft_center  = LLane.x_base[-1]
        win_xright_center = RLane.x_base[-1]
        lane_distance     = RLane.x_base[-1] - LLane.x_base[-1]
    
    lane_distance = win_xright_center - win_xleft_center
    
    ##------------------------------------------------------------------------------------------------
    ## Reset Pixels per Meter conversion factor to match detected lanes 
    ##------------------------------------------------------------------------------------------------
    # LLane.set_MX(LLane.MX_nom, rightx_base - leftx_base, debug = False)
    # RLane.set_MX(RLane.MX_nom, rightx_base - leftx_base, debug = False)
    # Identify the x and y positions of all nonzero pixels in the image

  
    # Current positions to be updated later for each window in nwindows
    
    # LLane.set_MX(denom = lane_distance)
    # RLane.set_MX(denom = lane_distance)

    
    if debug:
        print(' reset search base            : ', reset_search_base)
        print(' histogram before padding     : {:6d}    After: {:6d}    Midpoint: {:6d}'.format(hg_shape_before, histogram.shape[0], midpoint)  )
        print(' histRange  :             Left: {:6d}    Right: {:6d}    Top: {:6d}    Bottom: {:6d}'.format(histLeft, histRight, histTop, histBot))
        print(' Prev frame x_base        Left: {:6d}    Right: {:6d}'.format(LLane.x_base[-1], RLane.x_base[-1]))
        print(' max x  from histogram    Left: {:6d}    Right: {:6d}'.format(np.max(histogram[:midpoint]),np.max(histogram[midpoint:])))
        print(' x_base from histogram    Left: {:6d}    Right: {:6d}'.format(np.argmax(histogram[:midpoint]),np.argmax(histogram[midpoint:]+midpoint)))
        print(' Search window center X - Left: {:6d}    Right: {:6d}'.format(win_xleft_center  , win_xright_center))
        print()
        print(' MX Left            : {:6.4f}  Right: {:6.4f} {:10s}   MY Left            : {:6.4f}    Right: {:6.4f}'.format(
                round(LLane.MX,4), round(RLane.MX,4), '', round(LLane.MY,4), round(RLane.MY,4)) )
        print(' MX_nominator Left  : {:6.2f}  Right: {:6.2f} {:10s}   MY_nominator Left  : {:6.2f}    Right: {:6.2f}'.format(
                round(LLane.MX,4), round(RLane.MX,4), '', LLane.MY_nom     , RLane.MY_nom))
        print(' MX_denominator Left: {:6.2f}  Right: {:6.2f} {:10s}   MY_denominator Left: {:6.2f}    Right: {:6.2f}'.format(
                LLane.MX_denom   , RLane.MX_denom   , '', LLane.MY_denom   , RLane.MY_denom))
        print()
        print('-'*140)
        print('|                 |               Left Sliding Windows                      |                  Right Sliding Windows                       |')
        print('| Win |  Y range  | Frm   cntr   To | X idxs | Y idxs | Pixels |  Cntr Chg  | Frm    cntr    To  | X idxs | Y idxs | Pixels |    Ctr Chg   |')
        print('-'*140)

    # Find the four below boundaries of the window ###
    win_xleft_low   = win_xleft_center  - Left_lower_margin   
    win_xleft_high  = win_xleft_center  + Left_higher_margin  
    win_xright_low  = win_xright_center - Right_lower_margin        
    win_xright_high = win_xright_center + Right_higher_margin  

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        # Draw the windows on the visualization image
        # print(win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high, wcolor, out_img.shape)
        
        wcolor = colors.to_rgba_array(window_color)[0] * 255
        cv2.rectangle(out_img, (win_xleft_low , win_y_low), (win_xleft_high , win_y_high), wcolor, 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), wcolor, 2) 
        
        ### MY SOLUTION: Identify the nonzero pixels in x and y within the window -------------
        left_x_inds = np.where((win_xleft_low <=  nonzerox) & (nonzerox < win_xleft_high))
        left_y_inds = np.where((win_y_low     <=  nonzeroy) & (nonzeroy < win_y_high))
        good_left_inds = np.intersect1d(left_x_inds,left_y_inds,assume_unique=False)
        
        right_x_inds = np.where((win_xright_low <= nonzerox) & (nonzerox < win_xright_high))
        right_y_inds = np.where((win_y_low     <=  nonzeroy) & (nonzeroy < win_y_high))
        good_right_inds = np.intersect1d(right_x_inds,right_y_inds,assume_unique=False)


        ###------------------------------------------------------------------------------------
        # UDACITY SOLUTION: Identify the nonzero pixels in x and y within the window ###
        #    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        #    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        #    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        #    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        ###------------------------------------------------------------------------------------

        # Append these indices to the lists
        left_line_inds.append(good_left_inds)
        right_line_inds.append(good_right_inds)
        
        old_win_xleft_ctr   = win_xleft_center
        old_win_xright_ctr  = win_xright_center
        old_win_xleft_low   = win_xleft_low  
        old_win_xleft_high  = win_xleft_high 
        old_win_xright_low  = win_xright_low 
        old_win_xright_high = win_xright_high        

        ### If #pixels found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position    ###

        if (maxpix >= good_left_inds.shape[0] > minpix):
            # leftx_current = int(nonzerox[good_left_inds].mean())
            win_xleft_center = int(np.median(nonzerox[good_left_inds]))
        else:
            pass 

        if (maxpix >= good_right_inds.shape[0] > minpix ) :
            # rightx_current = int(nonzerox[good_right_inds].mean())
            win_xright_center = int(np.median(nonzerox[good_right_inds]))
        else:
            pass

        left_msg  = '{:4d} > {:4d} '.format( old_win_xleft_ctr,  win_xleft_center)
        right_msg = '{:5d} > {:5d} '.format(old_win_xright_ctr, win_xright_center)

        if debug:
            print('| {:3d} | {:3d} - {:3d} | {:3d} - {:3d} - {:3d} | {:6d} | {:6d} | {:6d} |{:12s}| {:4d} - {:4d} - {:4d} |'\
                  ' {:6d} | {:6d} | {:6d} |{:14s}|'.format(
                    window, win_y_low, win_y_high,  
                    win_xleft_low , old_win_xleft_ctr , win_xleft_high, 
                    left_x_inds[0].shape[0], left_y_inds[0].shape[0], good_left_inds.shape[0], left_msg,
                    win_xright_low, old_win_xright_ctr, win_xright_high, 
                    right_x_inds[0].shape[0], right_y_inds[0].shape[0], good_right_inds.shape[0], right_msg))
        
        shift_amount = 15

        win_xleft_low   = win_xleft_center  -  Left_lower_margin 
        win_xleft_high  = win_xleft_center  +  Left_higher_margin
        win_xright_low  = win_xright_center -  Right_lower_margin      
        win_xright_high = win_xright_center +  Right_higher_margin            

        if debug2:
            print(' Left Window       | {:4d} - {:4d} - {:4d} |            Next Left Window   | {:4d} - {:4d} - {:4d} |'.format( 
                    old_win_xleft_low , old_win_xleft_ctr ,  old_win_xleft_high, win_xleft_low , win_xleft_center ,  win_xleft_high))
            print(' Right Window      | {:4d} - {:4d} - {:4d} |            Next Right Window  | {:4d} - {:4d} - {:4d} |'.format(
                    old_win_xright_low, old_win_xright_ctr, old_win_xright_high, win_xright_low, win_xright_center, win_xright_high))
            print()

    if debug:
        print('-'*140)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_line_inds  = np.concatenate(left_line_inds)
    right_line_inds = np.concatenate(right_line_inds)
    LLane.set_linePixels(nonzerox[left_line_inds], nonzeroy[left_line_inds])
    RLane.set_linePixels(nonzerox[right_line_inds], nonzeroy[right_line_inds])
    
    ttlImageNZPixels = nonzerox.shape[0]    
    LLane.pixelCount.append( left_line_inds.shape[0])
    RLane.pixelCount.append(right_line_inds.shape[0])    
    LLane.pixelRatio.append(round(LLane.pixelCount[-1]*100 / (laneSearchPixels) , 2))
    RLane.pixelRatio.append(round(RLane.pixelCount[-1]*100 / (laneSearchPixels) , 2))

    ttlLaneNZPixels  = LLane.pixelCount[-1] + RLane.pixelCount[-1]
    
    imgPixelRatio    = round( ttlImageNZPixels*100 / ttlImagePixels , 2)
    NztoSrchNzRatio  = round( ttlLaneNZPixels*100  / ttlSearchPixels, 2)
    NztoImageNzRatio = round( ttlLaneNZPixels*100  / (ttlImageNZPixels + 1) , 2)    

        
    if debug:
    # if True:
        print()
        print('sliding _window_detection() - cont''d' )
        print('-'*20)
        print('   Window search margin: ', search_margin)        
        print('   ttl pixels in image          : {:8d}    NZ pixels in image         : {:8d}   imgPixelRatio   : %{:8.2f}'.format(
                  ttlImagePixels, ttlImageNZPixels,  imgPixelRatio )) 
        print('   ttl pixels in search region  : {:8d}    NZ pixels in search region : {:8d}   NZtoSrchNZRatio : %{:8.2f}'.format(
                  ttlSearchPixels, ttlLaneNZPixels,  NztoSrchNzRatio ))
        print('   ttl non-zero pixels in image : {:8d}    NZ pixels in search region : {:8d}   NztoImageNzRatio: %{:8.2f}'.format(
                  ttlImageNZPixels, ttlLaneNZPixels, NztoImageNzRatio ))
        print('   Detected Pixel Count    Left : {:8d}    Right: {:8d} '.format(LLane.pixelCount[-1], RLane.pixelCount[-1]))    
        print('   Detected Pixel Ratio    Left : {:8.2f}  Right: {:8.2f}'.format(
                  LLane.pixelRatio[-1], RLane.pixelRatio[-1]))

    return  out_img, histogram, (imgPixelRatio, NztoSrchNzRatio, NztoImageNzRatio, ttlImageNZPixels, ttlLaneNZPixels)    


def polynomial_proximity_detection(binary_warped, LLane, RLane, **kwargs):
    '''
    lane pixel detection - alternative to sliding _window_detection()
    
    HYPERPARAMETER
    search_margin : width of the margin around the previous polynomial to search
                    default is POLY_SEARCH_MARGIN
    
    '''
    
    debug          = kwargs.get('debug'          ,  False)  
    debug2         = kwargs.get('debug2'         ,  False)  
    histWidthRange = kwargs.get('histWidthRange',  binary_warped.shape[1])      
    search_margin  = kwargs.get('search_margin'  ,  POLY_SEARCH_MARGIN)
    
    img_height = binary_warped.shape[0]
    img_width  = binary_warped.shape[1]
    out_img    = np.dstack((binary_warped, binary_warped, binary_warped, np.ones_like(binary_warped)))*255
    
    ttlImagePixels = img_height * img_width
    laneSearchPixels = (2 * search_margin) * img_height
    ttlSearchPixels = 2 * laneSearchPixels

    # Take a histogram of the bottom half of the image
    histLeft  = max(img_width//2 - histWidthRange, 0)  
    histRight = min(img_width//2 + histWidthRange, img_width)  
    histTop   = 2*img_height // 3  ## min(histDepthRange, img_height)
    histBot   = LLane.y_src_bot 
    histogram = np.sum(binary_warped[histTop : histBot, histLeft : histRight], axis=0)
    hg_shape_before = histogram.shape[0]

    histogram = np.pad(histogram, (histLeft, binary_warped.shape[1]-histRight))
    midpoint  = np.int(histogram.shape[0]//2)

    leftx_base  = np.argmax(histogram[:midpoint]) 
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint     
    
    if debug:
        print('polynomial_proximity_detection()')
        print('-'*20)
        print('   Search margin       : {}     '.format(search_margin))
        print('   Histogram max   Left: {}     Right: {}'.format(leftx_base, rightx_base))
        print('   Prev-1   X Base Left: {}     Right: {}'.format(LLane.x_base[-2], RLane.x_base[-2]))
        print('   Previous X Base Left: {}     Right: {}'.format(LLane.x_base[-1], RLane.x_base[-1]))
        
    # Grab activated pixels
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ### Set the area of search based on activated x-values 
    ### within the +/- margin of our polynomial function 
    fitted_x_right  = np.polyval(RLane.best_fit, nonzeroy) 
    fitted_x_left   = np.polyval(LLane.best_fit, nonzeroy) 

    left_line_inds  = ( (nonzerox > ( fitted_x_left - search_margin )) & (nonzerox < (fitted_x_left + search_margin)) ).nonzero()
    right_line_inds = ( (nonzerox > (fitted_x_right - search_margin)) & (nonzerox < (fitted_x_right + search_margin)) ).nonzero()

    # Extract left and right line pixel positions
    LLane.set_linePixels(nonzerox [left_line_inds], nonzeroy[left_line_inds])
    RLane.set_linePixels(nonzerox[right_line_inds], nonzeroy[right_line_inds])

    ttlImageNZPixels = nonzerox.shape[0]    
    LLane.pixelCount.append( left_line_inds[0].shape[0])
    RLane.pixelCount.append(right_line_inds[0].shape[0])    
    LLane.pixelRatio.append(round(LLane.pixelCount[-1]*100 / (laneSearchPixels) , 2))
    RLane.pixelRatio.append(round(RLane.pixelCount[-1]*100 / (laneSearchPixels) , 2))

    # LLane.LnNztoSRPixelRatio = round(LLane.pixelCount[-1]*100 / (ttlSearchPixels) , 4)
    # RLane.LnNztoSRPixelRatio = round(RLane.pixelCount[-1]*100 / (ttlSearchPixels) , 4)

    ttlLaneNZPixels   = LLane.pixelCount[-1] + RLane.pixelCount[-1]
    imgPixelRatio     = round( ttlImageNZPixels*100 / ttlImagePixels , 2)
    NztoSrchNzRatio   = round( ttlLaneNZPixels*100  / ttlSearchPixels, 2)
    NztoImageNzRatio  = round( ttlLaneNZPixels*100  / (ttlImageNZPixels + 1) , 2)    


    out_img  = displayPolySearchRegion(out_img, LLane, RLane, search_margin = search_margin, debug = debug2)      


    if debug:
        print()
        print('   ttl pixels in image          : {:8d}    NZ pixels in image         : {:8d}   imgPixelRatio   : %{:8.2f}'.format(
                  ttlImagePixels, ttlImageNZPixels,  imgPixelRatio ))
        print('   ttl pixels in search region  : {:8d}    NZ pixels in search region : {:8d}   NztoSrchNzRatio : %{:8.2f}'.format(
                  ttlSearchPixels, ttlLaneNZPixels,  NztoSrchNzRatio))  ##  ttlLaneNZPixels*100/ttlSearchPixels))
        print('   ttl non-zero pixels in image : {:8d}    NZ pixels in search region : {:8d}   NztoImageNzRatio: %{:8.2f}'.format(
                  ttlImageNZPixels, ttlLaneNZPixels, NztoImageNzRatio ))  ## ttlLaneNZPixels*100/ttlImageNZPixels))
        print('   Detected Pixel Count    Left : {:8d}    Right: {:8d} '.format(LLane.pixelCount[-1], RLane.pixelCount[-1]))    
        print('   Detected Pixel Ratio    Left : {:8.2f}    Right: {:8.2f} '.format(
                  LLane.pixelRatio[-1], RLane.pixelRatio[-1]))

    return  out_img, histogram, (imgPixelRatio, NztoSrchNzRatio, NztoImageNzRatio, ttlImageNZPixels, ttlLaneNZPixels)




