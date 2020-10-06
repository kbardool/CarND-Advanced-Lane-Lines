import sys, os, pprint, pickle
import cv2
import numpy as np
import matplotlib.image as mpimg
from   matplotlib import colors
from common.sobel           import apply_image_thresholds, perspectiveTransform
from common.utils           import (display_one, display_two, displayRoILines, displayText, displayGuidelines)
# from common.utils           import (sliding_window_detection_v1, colorLanePixels_v1, fit_polynomial_v1, displayPolynomial_v1,
#                                     plot_polynomial_v1 , displayDetectedRegion_v1, curvatureMsg_v1, offCenterMsg_v1)
                               
from classes.plotdisplay import PlotDisplay
pp = pprint.PrettyPrinter(indent=2, width=100)


class ImagePipeline(object):

    def __init__(self, cameraConfig, **kwargs):
        assert cameraConfig   is not None, ' Camera object must be specified'
        self.camera                     = cameraConfig
        self.height                     = self.camera.height
        self.width                      = self.camera.width
        self.camera_x                   = self.camera.width //2
        self.camera_y                   = self.camera.height
    
    def __call__(self, filename = None, mode = 0 , **kwargs):
        assert filename is not None, ' Filename must be specified' 
        assert 0 < mode < 3        , ' mode must be 1 or 2'

        # pp.pprint(kwargs)
        # print('-'*30)    
        print('\n Pipeline Input Parms : ')
        print('-'*30)
        img_filename_ext = os.path.basename(filename)
        img_filename = os.path.splitext(img_filename_ext)
        print(' Input image: ', img_filename_ext)

        imgInput = mpimg.imread(filename)
        height          = imgInput.shape[0]
        width           = imgInput.shape[1]
        
        print(' height:  {}    width:  {}    camera_x:  {}    camera_y: {}'.format(height, width, self.camera_x, self.camera_y))
        
        mode            = kwargs.get('mode'   , 1)
        debug           = kwargs.get('debug'  , False)
        debug2          = kwargs.get('debug2' , False)
        debug3          = kwargs.get('debug3' , False)
        displayResults  = kwargs.get('displayResults' , False)
        frameTitle      = kwargs.get('frameTitle'     , '')
        thresholdKey    = kwargs.get('thresholdKey'   , 'cmb_rgb_sat_mag_x')
        nwindows        = kwargs.get('nwindows'       ,  12)    
        window_margin   = kwargs.get('window_margin'  , 40)    

        ksize           = kwargs.get('ksize'          ,  7)
        x_thr           = kwargs.get('x_thr'          , (30,110))
        y_thr           = kwargs.get('y_thr'          , (30,110))
        mag_thr         = kwargs.get('mag_thr'        , (65,255))
        dir_thr         = kwargs.get('dir_thr'        , (40,65))
        rgb_thr         = kwargs.get('rgb_thr'        , (210,255))
        # sat_thr       = kwargs.get('sat_thr'        , (130,255))
        lvl_thr         = kwargs.get('lvl_thr'        , (195,255))
        sat_thr         = kwargs.get('sat_thr'        , (200,255))

        x_thr2          = x_thr
        y_thr2          = None
        mag_thr2        = (50,255) 
        dir_thr2        = (0,10)
        sat_thr2        = (80, 255)     
        
        ## Source/Dest points for Perspective Transform   
        x_src_top_left  = kwargs.get('x_src_top_left' ,   570)  ## 580 -> 573
        x_src_top_right = kwargs.get('x_src_top_right',   714)
        x_src_bot_left  = kwargs.get('x_src_bot_left' ,   220) 
        x_src_bot_right = kwargs.get('x_src_bot_right',  1090) 
        
        y_src_bot       = kwargs.get('y_src_bot'      ,   700)  ## image.shape[0] - 20
        y_src_top       = kwargs.get('y_src_top'      ,   465)  ## 460 -> 465 y_src_bot - 255

        x_dst_left      = kwargs.get('x_dst_left'     ,   300)
        x_dst_right     = kwargs.get('x_dst_right'    ,  1000)
        y_dst_top       = kwargs.get('y_dst_top'      ,     0)
        y_dst_bot       = kwargs.get('y_dst_bot'      ,  height - 1)    
        
        src_points_list = [(x_src_top_left , y_src_top),
                           (x_src_top_right, y_src_top),
                           (x_src_bot_right, y_src_bot),      
                           (x_src_bot_left , y_src_bot)]
                                
        src_points = np.array( src_points_list, dtype = np.float32)  

        dst_points_list = [ (x_dst_left , y_dst_top), 
                            (x_dst_right, y_dst_top), 
                            (x_dst_right, y_dst_bot), 
                            (x_dst_left , y_dst_bot)]
                                
        dst_points = np.array( dst_points_list, dtype = np.float32)

        ##----------------------------------------------------------------------------
        ## Source/Dest points for Perspective Transform  (FROM VIDEO PIPELINE)      
        # x_src_top_left  = kwargs.get('x_src_top_left' ,  600)
        # x_src_top_right = kwargs.get('x_src_top_right',  740)
        # x_src_bot_left  = kwargs.get('x_src_bot_left' ,  295)
        # x_src_bot_right = kwargs.get('x_src_bot_right', 1105)
        # y_src_top       = kwargs.get('y_src_bot'      ,  480)
        # y_src_bot       = kwargs.get('y_src_top'      ,  700)  

        # x_dst_left      = kwargs.get('x_dst_left'     ,   300)
        # x_dst_right     = kwargs.get('x_dst_right'    ,  1000)
        # y_dst_top       = kwargs.get('y_dst_top'      ,     0)
        # y_dst_bot       = kwargs.get('y_dst_bot'      , height - 1)    
        ##----------------------------------------------------------------------------

        ##----------------------------------------------------------------------------
        # ksize = 19 # Choose a larger odd number to smooth gradient measurements
        # grad_x_thr = (70,100)
        # grad_y_thr = (80,155)
        # mag_thr    = (90,160)    
        # theta1     = 45
        # theta2     = 67
        # sat_thr    = (90,255)
        ##----------------------------------------------------------------------------

        ##----------------------------------------------------------------------------
        # RoI_x_top_left  = x_src_top_left  ## - 3
        # RoI_x_top_right = x_src_top_right ## + 3
        # RoI_x_bot_left  = x_src_bot_left  ## - 3
        # RoI_x_bot_right = x_src_bot_right ## + 3
        # RoI_y_bot       = y_src_bot       ## + 3
        # RoI_y_top       = y_src_top       ## - 3
        #
        # print(' Y bottom: ', RoI_y_bot, '   y_top : ', RoI_y_top)
        # RoI_vertices_list = [(RoI_x_bot_left , RoI_y_bot), 
        #                      (RoI_x_top_left , RoI_y_top), 
        #                      (RoI_x_top_right, RoI_y_top), 
        #                      (RoI_x_bot_right, RoI_y_bot)]
        # RoI_vertices      = np.array([RoI_vertices_list],dtype = np.int32)
        ##----------------------------------------------------------------------------

        ##----------------------------------------------------------------------------
        # warpedRoIVertices_list = [(x_transform_left-2 , 0), 
        #                           (x_transform_right+2, 0), 
        #                           (x_transform_right+2, height-1), 
        #                           (x_transform_left-2 , height-1)]
        # warpedRoIVertices = np.array([warpedRoIVertices_list],dtype = np.int32)
        # src_points = np.array([ ( 611, RoI_y_top), 
        #                         ( 666, RoI_y_top), 
        #                         (1055, RoI_y_bot), 
        #                          (250, RoI_y_bot)],dtype = np.float32)
        # src_points = np.array([ ( 568, RoI_y_top), 
        #                         ( 723, RoI_y_top), 
        #                         (1090, RoI_y_bot), 
        #                         ( 215, RoI_y_bot)],dtype = np.float32)
        # dst_points = np.array([ (x_transform_left,    2), 
        #                         (x_transform_right,   2), 
        #                         (x_transform_right, 718), 
        #                         (x_transform_left,  718)],dtype = np.float32)
        # src_points = np.array([ ( 595, RoI_y_top),
        #                         ( 690, RoI_y_top),
        #                         (1087, RoI_y_bot),         ### was 692
        #                         ( 228, RoI_y_bot)],dtype = np.float32)   ### was 705
        ##----------------------------------------------------------------------------
        
        ##----------------------------------------------------------------------------
        ## Perspective Transform Source/Dest points
        #
        # straight_line1_transform_points = {
        # 'x_src_top_left'  :  575,
        # 'x_src_top_right' :  708,
        #
        # 'x_src_bot_left'  :  220,  ## OR 230 
        # 'x_src_bot_right' : 1090,  ## OR 1100
        #
        # 'y_src_bot'       :  700 , ## image.shape[0] - 20
        # 'y_src_top'       :  460 , ## y_src_bot - 255
        #
        # 'x_dst_left'      : 300,
        # 'x_dst_right'     : 1000,
        # 'y_dst_top'       : 0,
        # 'y_dst_bot'       : height - 1
        # }
        #
        ##----------------------------------------------------------------------------

        if not displayResults:
            print(' Display results is fasle!')

        ###----------------------------------------------------------------------------------------------
        ###  Remove camera distortion and apply perspective transformation
        ###----------------------------------------------------------------------------------------------
        imgUndist = self.camera.undistortImage(imgInput)

        imgWarped, M, Minv = perspectiveTransform(imgUndist, src_points, dst_points, debug = False)


        ##----------------------------------------------------------------------------
        ## Image Tresholding
        ##----------------------------------------------------------------------------
        imgThrshldDict = apply_image_thresholds(imgUndist, ksize=ksize, 
                                            x_thr = x_thr, y_thr = y_thr, 
                                        mag_thr = mag_thr, dir_thr = dir_thr,  rgb_thr = rgb_thr,
                                        lvl_thr = lvl_thr, sat_thr = sat_thr, debug = debug2)
        
        imgThrshld = imgThrshldDict[thresholdKey]  

        ##----------------------------------------------------------------------------
        ## Apply Persepective Transformation
        ##----------------------------------------------------------------------------
        imgThrshldWarped, M, Minv = perspectiveTransform(imgThrshld, src_points, dst_points, debug = True)



        ###----------------------------------------------------------------------------------------------
        ### Display thresholded image before and after Perspective transform WITH RoI line display
        ###----------------------------------------------------------------------------------------------
        # imgThrshldRoI = displayRoILines(imgThrshld, RoI_vertices_list, thickness = 1)
        # imgThrshldRoIWarped, M, Minv = perspectiveTransform(imgThrshldRoI, src_points, dst_points, debug = False)
        # print('imgThrshldRoI shape     :', imgThrshldRoI.shape, imgThrshldRoI.min(), imgThrshldRoI.max())
        # print('img Thrshld Warped shape:', imgThrshldWarped.shape, imgThrshldWarped.min(), imgThrshldWarped.max())
        # display_two(imgThrshldRoI, imgThrshldRoIWarped, title1 = 'imgThrshldRoI',title2 = 'imgThrshldRoIWarped', winttl = filename[0])

        ###----------------------------------------------------------------------------------------------
        ### Display thresholded image without and with RoI line display
        ###----------------------------------------------------------------------------------------------
        # display_two(imgThrshld, imgThrshldRoI, title1 = 'imgThrshld',title2 = 'imgThrshldRoI', winttl = filename[0])

        ###----------------------------------------------------------------------------------------------
        ### Display MASKED color image with non RoI regions masked out -- With RoI line display
        ###----------------------------------------------------------------------------------------------
        # imgMaskedDbg = region_of_interest(imgRoI, RoI_vertices)
        # imgMaskedWarpedDbg, _, _ = perspectiveTransform(imgMaskedDbg, src_points, dst_points, debug = False)
        # print('imgMaskedDebug shape    :', imgMaskedDbg.shape, imgMaskedDbg.min(), imgMaskedDbg.max())
        # print('imgMaskedWarpedDbg shape:', imgMaskedWarpedDbg.shape, imgMaskedWarpedDbg.min(), imgMaskedWarpedDbg.max())
        # display_two(imgMaskedDbg  , imgMaskedWarpedDbg, title1 = 'imgMaskedDebug',title2 = ' imgMaskedWarpedDebug', winttl = filename[0])

        ###----------------------------------------------------------------------------------------------
        ### Display MASKED color image with non RoI regions masked out -- WITHOUT RoI line display
        ###----------------------------------------------------------------------------------------------
        # imgMaskedDbg = region_of_interest(imgUndist, RoI_vertices)
        # imgMaskedWarpedDbg, _, _ = perspectiveTransform(imgMaskedDbg, src_points, dst_points, debug = False)
        # print('imgMaskedDebug shape    :', imgMaskedDbg.shape, imgMaskedDbg.min(), imgMaskedDbg.max())
        # print('imgMaskedWarpedDbg shape:', imgMaskedWarpedDbg.shape, imgMaskedWarpedDbg.min(), imgMaskedWarpedDbg.max())
        # display_two(imgMaskedDbg  , imgMaskedWarpedDbg, title1 = 'imgMaskedDebug',title2 = ' imgMaskedWarpedDebug', winttl = filename[0])

        ###----------------------------------------------------------------------------------------------
        ### Mode 2: 
        ###  Warp image first, then apply thresholding 
        ###----------------------------------------------------------------------------------------------
        imgWarpedThrshld  = apply_image_thresholds(imgWarped, ksize=ksize, ret = thresholdKey,
                                                    x_thr = x_thr2 , y_thr   = y_thr2, 
                                                    mag_thr = mag_thr, dir_thr = dir_thr,
                                                    sat_thr = sat_thr, debug   = False)

        # imgWarpedThrshld = imgWarpedThrshldDict[thresholdKey]

        ################################################################################################
        ### Select image we want to process further 
        ################################################################################################
        if mode == 1:
            wrk_title = ' Mode 1: imgThrshldWarped : Threshold --> Warp ' 
            working_image = imgThrshldWarped; sfx = '_thr_wrp'   ### Warped AFTER thresholding
        else:
            wrk_title = ' Mode 2: imgWarpedThrshld : Warp --> Thresholding '
            working_image = imgWarpedThrshld; sfx = '_wrp_thr'   ### Warped BEFORE thresholding

        # display_one(working_image, title = wrk_title)


        ###----------------------------------------------------------------------------------------------
        ### Display thresholded image befoire and after Perspective transform WITHOUT RoI line display
        ### Display undistorted color image & perpective transformed image -- WITHOUT RoI line display
        ###----------------------------------------------------------------------------------------------
        if displayResults:
            ###----------------------------------------------------------------------------------------------
            ### Display undistorted color image & perpective transformed image -- With RoI line display
            ###----------------------------------------------------------------------------------------------
            imgRoI             = displayRoILines(imgUndist, src_points_list, thickness = 2)
            imgRoIWarped, _, _ = perspectiveTransform(imgRoI , src_points, dst_points, debug = False)
            imgRoIWarped       = displayRoILines(imgRoIWarped, dst_points_list, thickness = 2, color = 'green')
            # print('imgRoI shape       :', imgRoI.shape, imgRoI.min(), imgRoI.max())
            # print('imgRoIWarped shape       :', imgRoIWarped.shape, imgRoIWarped.min(), imgRoIWarped.max())
            # print('imgUndist shape       :', imgUndist.shape, imgUndist.min(), imgUndist.max())
            # print('imgWarped shape       :', imgWarped.shape, imgWarped.min(), imgWarped.max())
            # print('img Thrshld Warped shape:', imgThrshldWarped.shape, imgThrshldWarped.min(), imgThrshldWarped.max())

            display_two(imgInput, imgUndist, 
                        title1 = 'image - Original Image',
                        title2 = 'imgUndist - Undistorted Image', winttl = filename[0])
            display_two(imgRoI  , imgRoIWarped, fontsize=17, 
                        title1 = 'imgRoI',
                        title2 = 'imgRoIWarped', winttl = filename[0])  
            display_two(imgUndist, imgThrshld, 
                        title1 = 'imgUndist - Undistorted Image',
                        title2 = 'imgThrshld - using '+thresholdKey)
                        # title2 = 'imgWarped - Undistorted and Perspective Transformed', winttl = filename[0])
            display_two(imgThrshld, imgThrshldWarped, 
                        title1 = 'imgThrshld - using '+thresholdKey,
                        title2 = 'imgThrshldWarped - Thresholded and Warped image ', winttl = filename[0])
            display_two(imgUndist  , imgWarped, 
                        title1 = 'imgUndist - Undistorted Image',
                        title2 = 'imgWarped - Undistorted and Perspective Transformed', winttl = filename[0])
            display_two(imgWarped, imgWarpedThrshld,
                        title1 = 'imgWarped - Warped Image',
                        title2 = 'imgWarpedThrshld - Warped, Thresholded image', winttl = filename[0])
            display_two( imgThrshldWarped, imgWarpedThrshld,
                        title1 = 'Mode 1 - Image Warped AFTER Thresholding',
                        title2 = 'Mode 2 - Image Warped BEFORE Thresholding')  

            # histograms on thresholded image used to find source x coordinates for lane detection 
            hist = np.sum(imgThrshldWarped[2*imgThrshldWarped.shape[0]//3:, :], axis=0)
            reg  =  imgThrshldWarped[2*imgThrshldWarped.shape[0]//3:, :]
            display_two( reg, hist, size = (25,5), 
                        title1 = ' Mode 1 - Image Warped AFTER Thresholding', 
                        title2 = ' Histogram of activated pixels') 
                                                                
            hist = np.sum(imgWarpedThrshld[2*imgWarpedThrshld.shape[0]//3:, :], axis=0)
            reg  =  imgWarpedThrshld[2*imgWarpedThrshld.shape[0]//3:, :]
            display_two( reg, hist, size = (25,5),  
                        title1 = ' Mode 2 - Image Warped BEFORE Thresholding',
                        title2 = ' Histogram of activated pixels') 
             
        ##----------------------------------------------------------------------------------------------
        ## lane pixel detection 
        ##----------------------------------------------------------------------------------------------
        leftx, lefty, rightx, righty, out_img, histogram = sliding_window_detection_v1(working_image,
                                                                               nwindows = nwindows,
                                                                               window_margin = window_margin,
                                                                               debug = debug2)
    
        # imgLanePixels = colorLanePixels_v1(out_img, leftx, lefty, rightx, righty)
        left_fit, right_fit = fit_polynomial_v1(leftx, lefty, rightx, righty)
        ploty, left_fitx, right_fitx = plot_polynomial_v1(height, left_fit, right_fit)


        curv_msg = curvatureMsg_v1(700, left_fit, right_fit, 'm', debug = False)[0]
        oc_msg   = offCenterMsg_v1(700, left_fitx[700], right_fitx[700], self.camera_x, debug = False)

        result = displayDetectedRegion_v1(imgUndist, left_fitx, right_fitx, Minv)
        result = displayText(result, 40,40, curv_msg, fontHeight = 25)
        result = displayText(result, 40,80, oc_msg, fontHeight = 25)
        result = displayGuidelines(result, draw = 'y')

        ##----------------------------------------------------------------------------------------------
        ## Generate images displaying step-by-step process 
        ##----------------------------------------------------------------------------------------------
        if displayResults:
            print(oc_msg)
            
            print(curv_msg)
            
            displayCurvatures_v1(left_fit, right_fit)
    
            imgLanePixels = colorLanePixels_v1(out_img, leftx, lefty, rightx, righty)
            imgLanePixelsFitted = displayPolynomial_v1(imgLanePixels, ploty, left_fitx, right_fitx)
            
            display_two(imgLanePixelsFitted, histogram , size=(25,7),
                        title1 = ' imgLanePixels: Warped image Lane Pixels',
                        title2 = ' Histogram of activated pixels')

            
            disp  = PlotDisplay(5,2)
            disp.addPlot(imgInput, title=' Original Image ')
            disp.addPlot(imgThrshld, title=' Thresholded Image ')

            disp.addPlot(imgRoI, title = 'Original Image - Undistorted')
            disp.addPlot(imgRoIWarped, title = ' Undistorted image after perspective transformation')

            disp.addPlot(working_image, title=' Thresholded / Warped Image')  ## same as imgWarped
            disp.addPlot(histogram ,type = 'plot' ,  title = ' Histogram of activated pixels')
            
            disp.addPlot(imgLanePixels, title = ' imgLanePixels: Detected Lane Pixels ')
            disp.addPlot(imgLanePixels)
            disp.addPlot(left_fitx    , ploty,  subplot = disp.subplot, type = 'plot', color = 'yellow')
            disp.addPlot(right_fitx   , ploty,  subplot = disp.subplot, type = 'plot', 
                         title = 'Detected Lane Pixels + fitted polynomials', color = 'yellow')

            disp.addPlot(imgInput, title=' Original Image ')
            disp.addPlot(result, title=' Result of Lane Detection Process')
            disp.closePlot()
        else:
            disp = None
            
        return result, disp


def sliding_window_detection_v1(binary_warped, histRange = None,
                        nwindows = 9, 
                        window_margin = 100, 
                        minpix   = 50,
                        maxpix   = 99999, 
                        debug    = False):
    ''' 
    Parameters :  
    
    nwindows :                    number of sliding windows per lane
    window_height :               height of windows - based on nwindows above and image shape
    margin :                      Set the width of the windows +/- margin
    minpix :                      minimum number of detected pixels required to recenter window
    maxpix :                      maximum number of detected pixels allowed  to recenter window
    '''
    window_height = np.int(binary_warped.shape[0]//nwindows) 
    
    if maxpix == 0 :
        maxpix = (window_height * window_margin)
        
        
    # LLane.set_height(binary_warped.shape[0])
    # RLane.set_height(binary_warped.shape[0])

    if histRange is None:
        histLeft = 0
        histRight = binary_warped.shape[1]
    else:
        histLeft, histRight = int(histRange[0]), int(histRange[1])
    
    # Create an blank, black RGBA image to draw on and visualize the result
    out_img = np.dstack((binary_warped , binary_warped, binary_warped,  np.ones_like(binary_warped)))
    out_img *= 255
    
    # Take a histogram of the bottom half of the image    
    histogram_before = np.sum(binary_warped[2*binary_warped.shape[0]//3:, histLeft:histRight], axis=0)
    histogram_after  = np.pad(histogram_before, (histLeft, binary_warped.shape[1]-histRight))

    if debug:
        display_two(binary_warped, out_img, title1 = 'binary_warped '+str(binary_warped.shape))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint    = np.int(histogram_after.shape[0]//2)
    leftx_base  = np.argmax(histogram_after[:midpoint]) 
    rightx_base = np.argmax(histogram_after[midpoint:]) + midpoint   

    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
  
    # Current positions to be updated later for each window in nwindows
    leftx_current  = leftx_base
    rightx_current = rightx_base
    
    if debug:
        print(' Run sliding _window_detection_v1()  - histRange:', histRange)
        print(' histogram shape before padding: ' , histogram_before.shape)
        print(' histogram shape after padding : ' , histogram_after.shape) 
        print(' Midpoint (histogram//2): {} '.format(midpoint))
        print(' Histogram left side max: {}  right side max: {}'.format(
               np.argmax(histogram_after[:midpoint]), np.argmax(histogram_after[midpoint:])))
        print(' Histogram left side max: {}  right side max: {}'.format(leftx_base, rightx_base))
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
        
        ### MY SOLUTION: Identify the nonzero pixels in x and y within each window -------------
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
        leftx  = nonzerox[left_line_inds] ; lefty  = nonzeroy[left_line_inds]
        rightx = nonzerox[right_line_inds]; righty = nonzeroy[right_line_inds]
        rc = 1
        
    if debug:
        print()
        print(' leftx : ', leftx.shape, ' lefty : ', lefty.shape)
        print(' rightx : ', rightx.shape, ' righty : ', righty.shape)

    return leftx, lefty, rightx, righty, out_img, histogram_after


def fit_polynomial_v1(leftx, lefty, rightx, righty, debug = False):
    '''
    fit polynomial on detected lane pixels 
    '''
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
    '''
    Generate x values for fitting polynomial for plotting
    '''
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


def displayPolynomial_v1(input_img, ploty, leftInput, rightInput, **kwargs):
    assert type(leftInput) == type(rightInput), 'Second and Third parms must have matching types'    
    '''
     display a Line.fitted_* plot line
    '''
    # print(' displayPolynomial(): ', input_img.shape)
    img_height, img_width  = input_img.shape[0:2]
    color        = kwargs.get('color', 'yellow') 
    start        = kwargs.get('start', 0)  
    end          = kwargs.get('end'  , img_width)     
    debug        = kwargs.get('debug', False)
    iteration    = kwargs.get('iteration', -1)
    thickness    = kwargs.get('thickness',  1)
    LLane  = leftInput
    RLane  = rightInput

    if debug:
        print(' displayPolynomial_v1()')
        print('     input img shape: ', input_img.shape, img_height, img_width, ' start: ', start, 'end: ', end )
        print('     leftInput: ', leftInput.shape, ' rightInput: ', rightInput.shape)

    left_idx  = (end > LLane ) & (LLane >= start) 
    right_idx = (end > RLane ) & (RLane >= start) 
    plot_y    = np.int_(ploty)
    left_x    = np.int_(LLane[left_idx])
    right_x   = np.int_(RLane[right_idx])
    # left_x    = np.clip(left_x  , 0, img_width-1)
    # right_x   = np.clip(right_x , 0, img_width-1)
        
    colorRGBA = colors.to_rgba_array(color)[0]*255    
    result = np.copy(input_img)
    
    for i in range(0,thickness,1):
        plot_left_x  = np.clip(left_x + thickness , 0, img_width - 1)
        plot_right_x = np.clip(right_x + thickness, 0, img_width - 1)
        result[ plot_y ,  plot_left_x ] = colorRGBA
        result[ plot_y ,  plot_right_x] = colorRGBA
    
    return  result 


def displayDetectedRegion_v1(input_img, left_fitx, right_fitx,  Minv, **kwargs ):
    ''' 
    Overlay area between detected lanes over undistorted image 

    Parameters
    ----------
    leftInput,     Left/Right lane objects or pixel array of fitted lines
    rightInput

    Minv
    LLane, RLane:  Either Line objects or *_fitx numpy arrays
    
    iteration: item from xfitted_history to use for lane region zoning
               -1 : most recent xfitted current_xfitted (==  xfitted_history[-1])
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
    Original version used for Image Pipeline

    Parameters:
    -----------
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


def calculate_radius(y_eval, fit_coeffs, units, MX_denom = 700, MY_denom = 720, debug = False):
    MY = 30/MY_denom # meters per pixel in y dimension
    MX= 3.7/MX_denom # meters per pixel in x dimension
    A,B,_ = fit_coeffs   
    if units == 'm':
        A = (A * MX)/ (MY**2)
        B = (B * MX/MY)
    
    return  ((1 + ((2*A*(y_eval*MY))+B)**2)** 1.5)/np.absolute(2*A)   


def displayCurvatures_v1(left_fit, right_fit):
    '''
    Display table of lane curvatures for visualization / debugging purposes
    '''
    print()
    print('radius of curvature'.center(80))
    print('-'*80)
    print("  {:8s}  {:8s}   {:8s}   {:8s}      {:8s}   {:8s}  {:8s} ".format(" y eval" ,"avg pxl", "left_pxl" , "right_pxl", "avg mtr","left_mtr", "right_mtr"))
    print('-'*80)
    for y_eval in range(20,730,50):
        msg, curv_avg_pxl, curv_left_pxl, curv_right_pxl = curvatureMsg_v1(y_eval, left_fit, right_fit, 'p')
        msg, curv_avg_mtr, curv_left_mtr, curv_right_mtr = curvatureMsg_v1(y_eval, left_fit, right_fit, 'm')
        print(" {:8.0f}   {:8.2f}   {:8.2f}   {:8.2f}      {:8.2f}   {:8.2f}   {:8.2f} ".format(y_eval, 
                curv_avg_pxl, curv_left_pxl, curv_right_pxl, 
                curv_avg_mtr, curv_left_mtr, curv_right_mtr))


def curvatureMsg_v1(y_eval, left_fit, right_fit, units = 'm', MX_denom = 700, MY_denom = 720, debug = False):
    '''
    Calculates the curvature of polynomial functions in pixels.
    y_eval:               y-value where we want radius of curvature
    left_fit, right_fit:  polynomial parameters
    '''
    assert units in ['m', 'p'], "Invalid units parameter, must be 'm' for meters or 'p' for pixels"
    str_units  = " m" if units == 'm' else " pxls"
    
    left_curve = calculate_radius(y_eval, left_fit , units = units) 
    right_curve= calculate_radius(y_eval, right_fit, units = units) 
    avg_curve  = round((left_curve + right_curve)/2,3)
    message = "Curvature  L: "+str(int(left_curve))+str_units+"  R: "+str(int(right_curve))+str_units
    
    if debug:
        print("  {:8s}  {:8s}   {:8s}   {:8s} ".format(" y eval" ,"avg pxl", "left_pxl" , "right_pxl"))
        print(" {:8.0f}   {:8.2f}   {:8.2f}   {:8.2f} ".format( y_eval, avg_curve, left_curve, right_curve))
        print(message)
    return  message, avg_curve, left_curve, right_curve


def colorLanePixels_v1(input_img, leftx, lefty, rightx, righty, lcolor = 'red', rcolor = 'blue', debug = False):
    '''
    Color left/right lane pixels detected in find_lane_pixels_v1() for visualization
    '''
    if debug: 
        print(' Call displayLanePixels')
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
    