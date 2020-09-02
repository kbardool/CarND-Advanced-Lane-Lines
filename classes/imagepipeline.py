import sys, os, pprint, pickle
import numpy as np
import matplotlib.image as mpimg
from common.sobel           import apply_image_thresholds, perspectiveTransform
from common.utils           import (display_one, display_two, displayRoILines, displayText,
                                    displayGuidelines, displayCurvatures)
from common.utils           import (sliding_window_detection_v1, colorLanePixels_v1, fit_polynomial_v1, displayPolynomial_v1,
                                    plot_polynomial_v1 , displayDetectedRegion_v1, curvatureMsg_v1, offCenterMsg_v1)
                               
from classes.plotting import PlotDisplay
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
            print(curv_msg)
            print(oc_msg)
            displayCurvatures(left_fit, right_fit)
    
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




