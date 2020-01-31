import os
import sys
if '..' not in sys.path:
    print("pipeline.py: appending '..' to sys.path")
    sys.path.append('..')
import numpy as np
import cv2
import matplotlib.image as mpimg
import pprint 
pp = pprint.PrettyPrinter(indent=2, width=100)


# pp.pprint(sys.modules)
print(os.getcwd())
pp.pprint(sys.path)


from classes.line import Line
from classes.plotting import PlotDisplay
from common.utils import (perspectiveTransform, find_lane_pixels  , search_around_poly, adjust_contrast, adjust_gamma, 
                          offCenterMsg        , curvatureMsg      , colorLanePixels   , displayPolynomial, displayRoILines,  
                          displayLaneRegion   , displayText       , displayGuidelines , displayPolySearchRegion)
from common.sobel import apply_thresholds

class ALFPipeline(object):
    HISTORY = 8
    HISTOGRAM_SEARCH_MARGIN  = 370
    WINDOW_SEARCH_MARGIN     = 75
    POLY_SEARCH_MARGIN       = 20
    NAME     = 'ALFConfig'
    units    = 'm'
    NWINDOWS = 12
    MINPIX   = 90
    MAXPIX   = 8000
    
    def __init__(self, cameraConfig, **kwargs):
        # ksize = 19 # Choose a larger odd number to smooth gradient measurements
        # grad_x_thr = (70,100)
        # grad_y_thr = (80,155)
        # mag_thr    = (90,160)
        # theta1     = 45
        # theta2     = 67
        # sat_thr    = (90,255)
       
        self.outputPath         = kwargs.setdefault('outputPath','./output_videos')
        self.debug              = kwargs.setdefault('debug',False)
        self.camera             = cameraConfig
        self.height             = self.camera.height
        self.width              = self.camera.width
        self.camera_x           = self.camera.width //2
        self.camera_y           = self.camera.height


        
        ## Image Thresholding params ---------------
        # self.ksize                = 7
        # self.grad_x_thr           = (30,255)
        # self.grad_y_thr           = (70,255)
        # self.mag_thr              = (35,255)
        # self.dir_thr              = (40,65)
        # self.sat_thr              = (65,255)
        # self.lvl_thr              = (180, 255)
        # self.rgb_thr              = (180,255)
        ## Image Thresholding params (1/28/20) ------
        self.ksize      = 7
        self.grad_x_thr = (30,255)
        self.grad_y_thr = (70,255)
        self.mag_thr    = (35,255)
        self.dir_thr    = (40,65)
        self.sat_thr    = (110,255)
        self.lvl_thr    = (205, 255)
        self.rgb_thr    = (205,255)
        ##---------------------------------------------
       
       
        ##  Threshold params for Warped Image ----------
        # self.ksize_2              = 7
        # self.grad_x_thr_2         = (30,255)
        # self.grad_y_thr_2         = (70,255)
        # self.mag_thr_2            = (10,50) 
        # self.dir_thr_2            = (5,25)  
        # self.sat_thr_2            = (70, 255)
        # self.lvl_thr_2            = (180, 255)
        # self.rgb_thr_2            = (180,255)
        
        ## Warped Image Threshold params (1/28/20) ------
        self.ksize_2      = 7
        self.grad_x_thr_2 = (30,255)
        self.grad_y_thr_2 = (70,255)
        ## Light Conditions ------------
        # self.mag_thr_2    = (10,50) 
        # self.dir_thr_2    = (0,30)  
        ## dark conditions--------------
        self.mag_thr_2    = (25,250) 
        self.dir_thr_2    = (0,30)
        #-------------------------------
        # self.sat_thr_2    = (70, 255)
        # self.lvl_thr_2    = (180, 255)
        # self.rgb_thr_2    = (180,255)
        self.sat_thr_2    = (110, 255)
        self.lvl_thr_2    = (205, 255)
        self.rgb_thr_2    = (205,255)
        ##------------------------------------------------

        ## Source/Dest points for Perspective Transform
        # self.x_src_top_left     =  575
        # self.x_src_top_right    =  708
        # self.x_src_bot_left     =  220 
        # self.x_src_bot_right    = 1090 
        # self.y_src_bot          =  700  ## image.shape[0] - 20
        # self.y_src_top          =  460  ## y_src_bot - 255
                                
        self.x_src_bot_left       =  300
        self.x_src_bot_right      = 1100
        self.x_src_top_left       =  600
        self.x_src_top_right      =  730

        self.y_src_top            =  480
        self.y_src_bot            =  700  
                                  
        self.x_dst_left           = 300
        self.x_dst_right          = 1000
        self.y_dst_top            = 0
        self.y_dst_bot            = self.height - 1
                                  
        self.curvature_y_eval     = self.y_src_bot
        self.offCenter_y_eval     = self.y_src_bot
                                  
        self.LeftLane             = Line(history = self.HISTORY, height = self.height)
        self.RightLane            = Line(history = self.HISTORY, height = self.height)
        
        print(' Pipeline Line init complete()')
        print(self.LeftLane.units,self.LeftLane.MX , self.LeftLane.MY)
        print(self.RightLane.units,self.RightLane.MX, self.RightLane.MY)        
        
        self.slidingWindowBootstrap  = False

        # print(' RoI Y bottom: ', self.y_src_bot, '   RoI y_top : ', self.y_src_top)
        self.src_points = np.array([ (self.x_src_top_left , self.y_src_top),
                                     (self.x_src_top_right, self.y_src_top),
                                     (self.x_src_bot_right, self.y_src_bot),      
                                     (self.x_src_bot_left , self.y_src_bot)],dtype = np.float32)  

        self.dst_points = np.array([ (self.x_dst_left , self.y_dst_top), 
                                     (self.x_dst_right, self.y_dst_top), 
                                     (self.x_dst_right, self.y_dst_bot), 
                                     (self.x_dst_left , self.y_dst_bot)],dtype = np.float32)        

        self.RoI_x_top_left    = self.x_src_top_left  ## - 3
        self.RoI_x_top_right   = self.x_src_top_right ## + 3
        self.RoI_x_bot_left    = self.x_src_bot_left  ## - 3
        self.RoI_x_bot_right   = self.x_src_bot_right ## + 3
        self.RoI_y_bot         = self.y_src_bot       ## + 3
        self.RoI_y_top         = self.y_src_top       ## - 3

        self.RoI_src_vertices_list = [(self.RoI_x_bot_left , self.RoI_y_bot), 
                                      (self.RoI_x_top_left , self.RoI_y_top), 
                                      (self.RoI_x_top_right, self.RoI_y_top), 
                                      (self.RoI_x_bot_right, self.RoI_y_bot)]

        self.RoI_src_vertices      = np.array([self.RoI_src_vertices_list],dtype = np.int32)


        self.RoI_dst_vertices_list = [(self.x_dst_left , self.y_dst_top), 
                                      (self.x_dst_right, self.y_dst_top), 
                                      (self.x_dst_right, self.y_dst_bot), 
                                      (self.x_dst_left , self.y_dst_bot)]

    def displayConfig(self):
        """Display Configuration values."""
        ttl = (self.NAME.upper() if self.NAME is not None else '') + " Configuration Parameters:"
        
        print()
        print(ttl)
        print("-"*len(ttl))
        
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
    def reset(self):
        self.slidingWindowBootstrap = False
        self.LeftLane               = Line(history = self.HISTORY, height = self.height)
        self.RightLane              = Line(history = self.HISTORY, height = self.height)
        self.LeftLane.set_MY(  30, self.y_src_bot - self.y_src_top)
        self.RightLane.set_MY( 30, self.y_src_bot - self.y_src_top)        
        return True
        
        
    def __call__(self, image, frameTitle = '' ):
        '''
        # compareKey = 'cmb_rgb_lvl_sat'
        # compareKey = 'cmb_mag_x'
          compareKey = 'cmb_rgb_lvl_sat_mag_x'

        '''
        mode   = kwargs.setdefault('mode'  , 1)
        debug  = kwargs.setdefault('debug' , False)
        debug2 = kwargs.setdefault('debug2', False)
        debug3 = kwargs.setdefault('debug3', False)
        frameTitle     = kwargs.setdefault('frameTitle', '')
        compareKey     = kwargs.setdefault('compareKey', 'cmb_rgb_lvl_sat_mag_x')
        displayResults = kwargs.setdefault('displayResults', False)
      
        imgUndist = self.camera.undistortImage(image)

        ################################################################################################
        # Experiment with using historgram on undistorted image to find source x coordinates for perspective 
        # Transformation. The detect points ususally correspond to points inside the lanes.
        # hist, reg, leftx_base, rightx_base = pixel_histogram(imgThrshld, 670, 700 , debug = False)
        # x_src_bot_left = leftx_base
        # x_src_bot_right = rightx_base
        # display_two(reg, hist,title1= ' Thrshld img - Bottom', title2 = ' Thrshld img - Bottom')
        # print( reg.shape, hist.shape)
        # print(' Left base: ', leftx_base, ' Right base:', rightx_base)
        ################################################################################################
        ### PIPELINE 
        imgGammaCorr   = adjust_gamma(imgUndist, gamma=2.5)
        imgContrastAdj = adjust_contrast(imgUndist, alpha = 1.7, beta = 5)
        img_to_process = imgUndist


        ###----------------------------------------------------------------------------------------------
        ### Select image we want to process further 
        ###----------------------------------------------------------------------------------------------
        if mode == 1:
            output1 = apply_thresholds(img_to_process,  
                                          # ret     = 'cmb_rgb_lvl_sat',
                                          ksize   = self.ksize, 
                                          x_thr   = self.grad_x_thr, 
                                          y_thr   = self.grad_y_thr, 
                                          mag_thr = self.mag_thr, 
                                          dir_thr = self.dir_thr, 
                                          sat_thr = self.sat_thr, 
                                          rgb_thr = self.rgb_thr,
                                          lvl_thr = self.lvl_thr,
                                          debug   = debug2)
            imgThrshld = output1[compareKey]
            imgThrshldWarped, _, Minv = perspectiveTransform(imgThrshld, self.src_points, self.dst_points, debug = debug3)
            working_image = imgThrshldWarped; sfx = '_thr_wrp'   ### Warped AFTER thresholding
        else:
            imgWarped, _ , Minv = perspectiveTransform(img_to_process, self.src_points, self.dst_points, debug = debug3)
            output2  = apply_thresholds(imgWarped, 
                                                 # ret     = 'cmb_rgb_lvl_sat',
                                                 ksize   = self.ksize_2, 
                                                 x_thr   = self.grad_x_thr_2, 
                                                 y_thr   = self.grad_y_thr_2, 
                                                 mag_thr = self.mag_thr_2, 
                                                 dir_thr = self.dir_thr_2, 
                                                 sat_thr = self.sat_thr_2, 
                                                 rgb_thr = self.rgb_thr_2,
                                                 lvl_thr = self.lvl_thr_2,
                                                 debug   = debug2)
            imgWarpedThrshld     = output2[compareKey]                   
            working_image = imgWarpedThrshld; sfx = '_wrp_thr'   ### Warped BEFORE thresholding
        
        ## DEBUG DISPLAYS
        if debug  and displayResults and mode == 2:
            imgWarpedThrshldLvl  = output2['cmb_lvl'] 
            imgWarpedThrshldSat  = output2['cmb_sat'] 
            imgWarpedThrshldDark = output2['cmb_mag_x']


            # display_two(image, imgUndist)
            # display_two(imgGammaCorr, imgContrastAdj)
            ttlT   = np.sum(imgThrshld)
            ttlWTS = np.sum(imgWarpedThrshldSat)
            ttlWTL = np.sum(imgWarpedThrshldLvl)
            ttlWT  = np.sum(imgWarpedThrshld)
            ttlWTD = np.sum(imgWarpedThrshldDark)
            display_two(imgUndist,  imgThrshld, title2 =  'Image Thresholded '+str(ttlT))
            display_two(imgWarped, imgWarpedThrshldLvl, title1 = 'Image warped', title2 = 'Warp -> Thrshld (Lvl)'+str(ttlWTL))
            display_two(imgWarpedThrshld, imgWarpedThrshldDark, title1 = 'Warp -> Thrshld (RGB,LVL,SAT)'+str(ttlWT), title2 = 'Warp -> Thrshld (MAG, X)'+str(ttlWTD))

        ###----------------------------------------------------------------------------------------------
        ### Find lane pixels 
        ###----------------------------------------------------------------------------------------------
        if self.slidingWindowBootstrap:
            rc, out_img, histogram = search_around_poly(working_image, self.LeftLane, self.RightLane, 
                                                    search_margin = self.POLY_SEARCH_MARGIN, 
                                                    debug = debug)
            out_img  = displayPolySearchRegion(out_img, self.LeftLane, self.RightLane, margin = self.POLY_SEARCH_MARGIN, debug = debug)                                                    
        else:    
            histRange = (self.camera_x - self.HISTOGRAM_SEARCH_MARGIN, self.camera_x + self.HISTOGRAM_SEARCH_MARGIN)
            
            rc, out_img, histogram = find_lane_pixels(working_image, self.LeftLane, self.RightLane, histRange, 
                                                  nwindows = self.NWINDOWS, 
                                                  window_margin = self.WINDOW_SEARCH_MARGIN, 
                                                  debug = debug)    
            self.slidingWindowBootstrap  = True
    
    
        ###----------------------------------------------------------------------------------------------
        ### Fit polynomial on found lane pixels 
        ###----------------------------------------------------------------------------------------------
        if rc:
            self.LeftLane.fit_polynomial(debug  = debug)
            self.RightLane.fit_polynomial(debug = debug)
        
        if debug and displayResults:
            print(frame_title)
            displayFittingInfo(LeftLane, RightLane)

            # imgLanePixels2 = displayPolynomial(out_img, LeftLane.best_xfitted_history, RightLane.best_xfitted_history, iteration = -2, color = 'black')
            display_one(histogram, size = 10)
            dbg = colorLanePixels(out_img, LeftLane, RightLane)
            # dbg = displayPolynomial(dbg, LeftLane.avg_xfitted, RightLane.avg_xfitted, color = 'black')
            dbg = displayPolynomial(dbg, LeftLane.best_xfitted, RightLane.best_xfitted, color = 'red')
            dbg = displayPolynomial(dbg, LeftLane.current_xfitted, RightLane.current_xfitted, iteration = -1, color = 'yellow', debug = True)
            display_one(dbg, title = frame_title)        
            
        acceptPolynomial = assessFittedPolynoms()
        
        if acceptPolynomial:
            LeftLane.acceptFittedPolynomial(debug  = debug)
            RightLane.acceptFittedPolynomial(debug = debug)
            print(frame_title + ' Accept propsed polynomials ' )            
        else:
            LeftLane.rejectFittedPolynomial(debug  = debug)
            RightLane.rejectFittedPolynomial(debug = debug)
            print(frame_title + ' propsed polynomials REJECTED ' )
            print(LeftLane.detected, RightLane.detected)
            print(LeftLane.framesSinceDetected, RightLane.framesSinceDetected)            

        if not(LeftLane.detected and RightLane.detected) and (LeftLane.framesSinceDetected > 8 or RightLane.framesSinceDetected > 8):
            print(' No detection and many non-detected frames')
            missingSequence = True
            polyRegionColor1 = 'red'
            polyRegionColor2 = 'red'
        else:
            print(' Good detection or less than 8 no-detection frames ')
            missingSequence = True
            polyRegionColor1 = 'green'
            polyRegionColor2 = 'yellow'

        curv_msg = curvatureMsg(self.LeftLane, self.RightLane, debug = debug)
        oc_msg   = offCenterMsg(self.LeftLane, self.RightLane, self.camera_x, debug = debug)

        region_zone_sep_pos = LeftLane.y_src_top
        print(LeftLane.current_fit, '  R: ', RightLane.current_fit)
        print(LeftLane.best_fit,    '  R: ', RightLane.best_fit)
        result_1 = displayLaneRegion(imgUndist, self.LeftLane.current_xfitted, self.RightLane.current_xfitted, 
                                     Minv, start=region_zone_sep_pos        , beta = 0.2, color = polyRegionColor1)
        result_1 = displayLaneRegion(result_1 , self.LeftLane.current_xfitted, self.RightLane.current_xfitted, 
                                     Minv, start=200,end=region_zone_sep_pos, beta = 0.2, color = polyRegionColor2)

        # result_1 = displayLaneRegion(imgUndist, self.LeftLane, self.RightLane, Minv)
        displayText(result_1, 40, 40, frameTitle, fontHeight = 20)
        displayText(result_1, 40, 80,   curv_msg, fontHeight = 20)
        displayText(result_1, 40,120,     oc_msg, fontHeight = 20)
        # displayGuidelines(result_1, draw = 'y');

        result_2 = displayLaneRegion(imgUndist, self.LeftLane.best_xfitted, self.RightLane.best_xfitted, 
                                     Minv, start=region_zone_sep_pos           , beta = 0.2, color = polyRegionColor1)
        result_2 = displayLaneRegion(result_2 , self.LeftLane.best_xfitted, self.RightLane.best_xfitted, 
                                      Minv, start = 200, end=region_zone_sep_pos, beta = 0.2 , color = polyRegionColor2)

        # result_2 = displayLaneRegion(imgUndist, self.LeftLane.best_xfitted, self.RightLane.best_xfitted, Minv)
        displayText(result_2, 40, 40, frameTitle, fontHeight = 20)
        displayText(result_2, 40, 80,   curv_msg, fontHeight = 20)
        displayText(result_2, 40,120,     oc_msg, fontHeight = 20)
        displayGuidelines(result_2, draw = 'y');
        
        
        if debug or displayResults:
            print(' Left lane MR fit      : ', LeftLane.current_fit , '    Right lane MR fit     : ', RightLane.current_fit)
            print(' Left lane MR best fit : ', LeftLane.best_fit    , '    Right lane MR best fit: ', RightLane.best_fit)
            print(' Left Curvature @ y =  10   : '+str(LeftLane.get_curvature(10))+" m   Right Curvature   : "+str(RightLane.get_curvature(10))+" m")
            print(' Left Curvature @ y = 700   : '+str(LeftLane.get_curvature(700))+" m   Right Curvature   : "+str(RightLane.get_curvature(700))+" m")
            print(' Curvature message : ', curv_msg)
            print(' Off Center Message: ', oc_msg)            

        if displayResults:
            ##----------------------------------------------------------------------------------------------
            ### Display undistorted color image & perpective transformed image -- With RoI line display
            ###----------------------------------------------------------------------------------------------
            imgRoI   = displayRoILines(img_to_process, self.RoI_src_vertices_list, thickness = 2)


            imgRoIWarped, _, _ = perspectiveTransform(imgRoI, self.src_points, self.dst_points)
            imgRoIWarped = displayRoILines(imgRoIWarped, self.RoI_dst_vertices_list, thickness = 2, color = 'green')
            # print('imgRoI shape       :', imgRoI.shape, imgRoI.min(), imgRoI.max())
            # display_one(imgRoI  , title = 'imgRoI -'+frame_title, winttl = frame_title, grid = 'major',size=size)
            # print('imgRoIWarped shape       :', imgRoIWarped.shape, imgRoIWarped.min(), imgRoIWarped.max())
            # display_one(imgRoIWarped  , title = 'imgRoIWarped -'+frame_title, winttl = frame_title, grid = 'major', size = size)


            ## Certain operations are not performed based on the processing mode selected
            ## Generate images for skipped operations for display purposes
            
            # out_img  = displayPolySearchRegion(out_img, self.LeftLane, self.RightLane, margin = self.POLY_SEARCH_MARGIN, debug = debug)
            imgLanePxls = colorLanePixels(out_img, self.LeftLane, self.RightLane)
            imgLanePxls = displayPolynomial(imgLanePxls, self.LeftLane.best_xfitted_history, self.RightLane.best_xfitted_history, 
                                            iteration = -2, color = 'black')
            imgLanePxls = displayPolynomial(imgLanePxls, self.LeftLane.best_xfitted        , self.RightLane.best_xfitted, color = 'red')
            imgLanePxls = displayPolynomial(imgLanePxls, self.LeftLane.current_xfitted     , self.RightLane.current_xfitted,
                                            iteration = -1, color = 'yellow', debug = True)


            # imgLanePxls = displayPolynomial(imgLanePxls, self.LeftLane.best_xfitted, self.RightLane.best_xfitted, color = 'red')
            # imgLanePxls = displayPolynomial(imgLanePxls, self.LeftLane, self.RightLane, iteration = -2, color = 'black')
            # imgLanePxls = displayPolynomial(imgLanePxls, self.LeftLane, self.RightLane, iteration = -1, color = 'yellow')

            if mode == 1:
                print('='*100)
                print(' Apply thresholding BEFORE warping')
                print('='*100)      
                imgWarped, _, _ = perspectiveTransform(imgUndist, self.src_points, self.dst_points, debug = debug3)
                output2  = apply_thresholds(imgWarped, 
                                         # ret     = 'cmb_rgb_lvl_sat',
                                         ksize   = self.ksize_2, 
                                         x_thr   = self.grad_x_thr_2, 
                                         y_thr   = self.grad_y_thr_2, 
                                         mag_thr = self.mag_thr_2, 
                                         dir_thr = self.dir_thr_2, 
                                         sat_thr = self.sat_thr_2, 
                                         rgb_thr = self.rgb_thr_2,
                                         lvl_thr = self.lvl_thr_2,
                                         debug   = debug2)
                imgWarpedThrshld = output2[compareKey]
            else: 
                print('='*100)
                print(' Apply thresholding AFTER warping')
                print('='*100)      
                output1 = apply_thresholds(imgUndist,  
                                              # ret     = 'cmb_rgb_lvl_sat',
                                              ksize   = self.ksize, 
                                              x_thr   = self.grad_x_thr, 
                                              y_thr   = self.grad_y_thr, 
                                              mag_thr = self.mag_thr, 
                                              dir_thr = self.dir_thr, 
                                              sat_thr = self.sat_thr, 
                                              rgb_thr = self.rgb_thr,
                                              lvl_thr = self.lvl_thr,
                                              debug   = debug2)
                imgThrshld = output1[compareKey]
                # imgThrshld = imgThrshldList['cmb_rgb_lvl_sat']
                imgThrshldWarped, _, _  = perspectiveTransform(imgThrshld, self.src_points, self.dst_points, debug = debug3) 
                
            ploty = np.linspace(0, self.height-1, self.height)

            disp = PlotDisplay(6,2)
            disp.addPlot(image           , title = 'original frame - '+frameTitle)
            disp.addPlot(imgUndist       , title = 'imgUndist - Undistorted Image')
            
            disp.addPlot(imgRoI          , title = 'imgRoI'   )
            disp.addPlot(imgRoIWarped    , title = 'imgRoIWarped' )
            
            disp.addPlot(imgThrshld      , title = 'imgThrshld - Thresholded image')
            disp.addPlot(imgWarped       , title = 'imgWarped - Warped Image')
            
            disp.addPlot(imgWarpedThrshld, title = 'imgWarpedThrshld - Image warped BEFORE Thresholding')
            disp.addPlot(imgThrshldWarped, title = 'imgThrshldWarped - Image warped AFTER Thresholding')
            
            disp.addPlot(imgLanePxls     , title = 'ImgLanePxls (Black: Prev fit, Yellow: New fit, Red: Best Fit)' )
            disp.addPlot(histogram       , title = 'Histogram of activated pixels', type = 'plot' )
            
            disp.addPlot(result_1        , title = 'result_1 : Using LAST fit')
            disp.addPlot(result_2        , title = 'result_2 : Using BEST fit'+frameTitle)
            disp.closePlot()
            return result_2, disp
        else:
            return result_2
            
            
    def displayPipelineInfo(self):            
        print()
        print(' Fit Polynomial Coeffs  ', '\n'+'-'*30)
        for i,j in zip(reversed(self.LeftLane.fit_history),reversed(self.RightLane.fit_history)):
            print('L: {}   R: {} '.format(i,j))

        print()
        print(' Current Fit ', '\n'+'-'*30)
        print('L: {}   R: {} '.format(self.LeftLane.current_fit ,self.RightLane.current_fit))    

        print()
        print(' Diffs ', '\n'+'-'*30)
        avg_i = 0; avg_j = 0
        for i,j in zip(reversed(self.LeftLane.fit_diffs_history),reversed(self.RightLane.fit_diffs_history)):
            avg_i += np.sqrt(np.sum(i**2))
            avg_j += np.sqrt(np.sum(j**2))
            print('L: {}     {:8.2f}      R: {}       {:8.2f}    '.format(i,np.sqrt(np.sum(i**2)), j,np.sqrt(np.sum(j**2))))
            
        print('Average         {:8.2f}                        {:8.2f}    '.format(avg_i/len(self.LeftLane.fit_diffs_history) , 
                                                                                     avg_j/len(self.RightLane.fit_diffs_history)))

        print()
        print(' Best Fit ', '\n'+'-'*30)
        print('L: {}   R: {} '.format(self.LeftLane.best_fit ,self.RightLane.best_fit))  

        print()
        print(' Best Fit History ', '\n'+'-'*30)
        for i,j in zip(reversed(self.LeftLane.best_fit_history),reversed(self.RightLane.best_fit_history)):
            print('L: {}   R: {} '.format(i ,j))    

        print()
        print(' Best RSE History ', '\n'+'-'*30)
        for i,j in zip(reversed(self.LeftLane.best_RSE_history),reversed(self.RightLane.best_RSE_history)):
            print('L: {:8.3f}   R: {:8.3f} '.format(i ,j))   
            
        print()
        avg_i = 0; avg_j = 0
        print(' Best Diffs History ', '\n'+'-'*30)
        for i,j in zip(reversed(self.LeftLane.best_diffs_history),reversed(self.RightLane.best_diffs_history)):
            avg_i += np.sqrt(np.sum(i**2))
            avg_j += np.sqrt(np.sum(j**2))
            print('L: {}     {:8.2f}      R: {}       {:8.2f}    '.format(i,np.sqrt(np.sum(i**2)), j,np.sqrt(np.sum(j**2))))
                 
        print()
        print(' Average Best Diffs ', '\n'+'-'*30)    
        print('L: {:8.2f}   R: {:8.2f} '.format(avg_i / len(self.LeftLane.best_diffs_history),avg_j / len(self.RightLane.best_diffs_history)))



        print()
        print(' Slope ', '\n'+'-'*30)
        for i,j, k,l in zip(reversed(self.LeftLane.slope) ,reversed(self.LeftLane.xfitted_history),
                            reversed(self.RightLane.slope),reversed(self.RightLane.xfitted_history)):
            print('L: {:8.2f}   R: {:8.2f}   {:8.2f}   R: {:8.2f} '.format(i,k, self.LeftLane.get_slope(700,j), self.RightLane.get_slope(700,l) ))
            
            
        print()
        avg_diff = 0
        print(' radius of curvature ', '\n'+'-'*30)
        for i,j ,k,l in zip(reversed(self.LeftLane.radius), reversed(self.LeftLane.fit_history),
                       reversed(self.RightLane.radius), reversed(self.RightLane.fit_history)):
            avg_diff += (i-j)
            l_curv    = self.LeftLane.get_curvature(700,j)
            r_curv    = self.RightLane.get_curvature(700,l)
            print('L: {:8.2f}   R: {:8.2f}   diff: {:8.2f}     {:8.2f}   {:8.2f}   {:8.2f}'.format(i,k, i-k,l_curv, r_curv, l_curv - r_curv)) 

            
        print()
        print(' Average radius of curvature ', '\n'+'-'*30)    
        print('L: {:8.2f}   R: {:8.2f}    Diff: {:8.2f} '.format(np.mean(self.LeftLane.radius), 
                                                                 np.mean(self.RightLane.radius), 
                                                                 avg_diff/len(self.LeftLane.radius)))

        print()
        print(' line_base_meters ', '\n'+'-'*30)
        for i,j in zip(reversed(self.LeftLane.line_base_meters),reversed(self.RightLane.line_base_meters)):
            print('L: {:8.2}   R: {:8.2f}   width : {:8.2f} '.format( i,  j,  round(j - i,2)))    
            
            
        print()
        print(' line_base_pixels ', '\n'+'-'*30)
        for i,j in zip(reversed(self.LeftLane.line_base_pixels),reversed(self.RightLane.line_base_pixels)):
            print('L: {:10.2f}   R: {:10.2f}   width : {:8.2f} '.format( i,  j,  round(j - i,2)))    
            
            
            
        print()
        sum_i = 0; sum_j = 0
        print(' history_xfitted @ Y = 700, 350, 0', '\n'+'-'*30)
        print('   Left    Right    Width         Left    Right    Width         Left    Right    Width    Left    Right    Width ')
        print('-'*130)
        for i,j in zip(reversed(self.LeftLane.xfitted_history),reversed(self.RightLane.xfitted_history)):
            sum_i += i[700]
            sum_j += j[700]
            print('{:7.2f}   {:7.2f}   {:7.2f}     {:7.2f}   {:7.2f}   {:7.2f}     {:7.2f}   {:7.2f}   {:7.2f}   {:7.2f}   {:7.2f}   {:7.2f}'.format(
                  i[700],  j[700], j[700]-i[700], i[600],  j[600], j[600]-i[600],
                  i[500],  j[500], j[500]-i[500], i[400],  j[400], j[400]-i[400]))   

        print()
        print(' Average history_xfitted @ Y = 700 ', '\n'+'-'*30)    
        print('L: {:8.2f}   R: {:8.2f}     Width: {:8.2f}'.format(sum_i/len(self.LeftLane.xfitted_history), sum_j /len(self.RightLane.xfitted_history),
                                                                   (sum_j - sum_i) / len(self.LeftLane.xfitted_history)))



        print('\n')
        print(' history_xfitted ', '\n'+'-'*30)
        for i,j in zip(reversed(self.LeftLane.xfitted_history),reversed(self.RightLane.xfitted_history)):
            print('L: {}   R: {}'.format( i[:5],  j[:5]))
        print()
        print(' current_xfitted ', '\n'+'-'*30)
        print('L: {}   R: {}    '.format(self.LeftLane.current_xfitted[:5],  self.RightLane.current_xfitted[:5]))       
                    
            
'''
    def find_lane_pixels(self, binary_warped, LLane, RLane, histRange = None,
                     nwindows = 9, 
                     window_margin   = 100, 
                     minpix   = 90,
                     maxpix   = 0, 
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
        
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, histLeft:histRight], axis=0)
        print(' histogram shape before padding: ' , histogram.shape)
        
        histogram = np.pad(histogram, (histLeft, binary_warped.shape[1]-histRight))
        print(' histogram shape after padding : ' , histogram.shape) 
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        
        midpoint    = np.int(histogram.shape[0]//2)
        leftx_base  = np.argmax(histogram[:midpoint]) 
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint     
        
        if debug:
            print(' Run find_lane_pixels()  - histRange:', histRange)
            print(' Midpoint:  {} '.format(midpoint))
            print(' Histogram left side max: {}  right side max: {}'.format(np.argmax(histogram[:midpoint]),np.argmax(histogram[midpoint:])))
            print(' Histogram left side max: {}  right side max: {}'.format(leftx_base, rightx_base))
        
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
            pass
        
        # Extract left and right line pixel positions
        LLane.set_linePixels(nonzerox[left_line_inds], nonzeroy[left_line_inds])
        RLane.set_linePixels(nonzerox[right_line_inds], nonzeroy[right_line_inds])
        
        if debug:
            print()
            print(' leftx : ', LLane.allx.shape, ' lefty : ', LLane.ally.shape)
            # print('   X:', LLane.allx[:15])
            # print('   Y:', LLane.ally[:15])
            # print('   X:', LLane.allx[-15:])
            # print('   Y:', LLane.ally[-15:])
            print(' rightx : ', RLane.allx.shape, ' righty : ', RLane.ally.shape)
            # print('   X:', RLane.allx[:15])
            # print('   Y:', RLane.ally[:15])
            # print('   X:', RLane.allx[-15:])
            # print('   Y:', RLane.ally[-15:])
            # display_one(out_img)

        return out_img, histogram 

'''

'''


    def search_around_poly(self, binary_warped, LLane, RLane, search_margin = 100, debug = False):
        """
        # HYPERPARAMETER
        # search_margin : width of the margin around the previous polynomial to search
        """
        out_img = np.dstack((binary_warped, binary_warped, binary_warped, np.ones_like(binary_warped)))*255

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
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
            
        left_line_inds  = ( (nonzerox > ( fitted_x_left - search_margin )) & (nonzerox < ( fitted_x_left + search_margin)) ).nonzero()
        right_line_inds = ( (nonzerox > (fitted_x_right - search_margin)) & (nonzerox <  (fitted_x_right + search_margin)) ).nonzero()
        
        if debug:
            print(' Search_around_poly() ')
            print(' fitted_x_left  : ', fitted_x_left.shape     , '  fitted_x_right : ', fitted_x_right.shape)
            print(' left_lane_inds : ',  left_line_inds[0].shape , left_line_inds)
            print(' right_lane_inds: ', right_line_inds[0].shape, right_line_inds)
        
        
        # Extract left and right line pixel positions
        # LLane.allx = nonzerox [left_line_inds]
        # LLane.ally = nonzeroy [left_line_inds] 
        # RLane.allx = nonzerox[right_line_inds]
        # RLane.ally = nonzeroy[right_line_inds]
        LLane.set_linePixels(nonzerox [left_line_inds],  nonzeroy[left_line_inds])
        RLane.set_linePixels(nonzerox[right_line_inds], nonzeroy[right_line_inds])
        
        # Fit new polynomials
        # left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        out_img= displayPolySearchRegion(out_img, LLane, RLane, debug = debug)
        
        return out_img, histogram
      
        # return result

'''

'''
    def find_lane_pixels_v1(self, binary_warped, histRange= None,  debug = False):
        self.LeftLane.height  = binary_warped.shape[0]
        self.RightLane.height = binary_warped.shape[0]

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        # if debug: 
            # print(' binary warped shape: ', binary_warped.shape)
            # print(' out_img shape: ', out_img.shape)
            # display_one(out_img, grayscale = False, title = 'out_img')
            # display_one(binary_warped, title='binary_warped')
        if histRange is None:
            histLeft = 0
            histRight = binary_warped.shape[1]
        else:
            histLeft, histRight = int(histRange[0]), int(histRange[1])

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,histLeft:histRight], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint    = np.int(histogram.shape[0]//2)
        leftx_base  = np.argmax(histogram[:midpoint]) + histLeft
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint + histLeft

        if debug:
            print(' Hist range left: {}  right: {} '.format(histLeft,histRight))
            print(' Midpoint:  {} '.format(midpoint))
            print(' Histogram left side max: {}  right side max: {}'.format(np.argmax(histogram[:midpoint]),np.argmax(histogram[midpoint:])))
            print(' Histogram left side max: {}  right side max: {}'.format(leftx_base, rightx_base))
        
        # HYPERPARAMETERS
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.NWINDOWS)
        # Set maximum number of pixels found to recenter window
        self.MAXPIX = (window_height * self.WINDOW_MARGIN)

        # Choose the number of sliding windows
        # NWINDOWS = 9
        # Set the width of the windows +/- margin
        # MARGIN = 100
        # Set minimum number of pixels found to recenter window
        # MINPIX = 90
        
        
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
        left_lane_inds  = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.NWINDOWS):
            # Identify window boundaries in x and y (and right and left)
            win_y_low  = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low   = leftx_current  - self.WINDOW_MARGIN # Update this
            win_xleft_high  = leftx_current  + self.WINDOW_MARGIN # Update this
            win_xright_low  = rightx_current - self.WINDOW_MARGIN # Update this     
            win_xright_high = rightx_current + self.WINDOW_MARGIN # Update this

            if debug:
                print()
                print(' Window: ', window, ' y range: ', win_y_low,' to ', win_y_high )
                print('-'*50)
                print(' Left  lane X range : ', win_xleft_low , '  to  ', win_xleft_high)
                print(' Right lane X range : ', win_xright_low, '  to  ', win_xright_high)
                
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low , win_y_low), (win_xleft_high , win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low, win_y_low), (win_xright_high, win_y_high),(0,255,0), 2) 
            
            ### MY SOLUTION: Identify the nonzero pixels in x and y within the window -------------
            left_x_inds = np.where((win_xleft_low <=  nonzerox) & (nonzerox < win_xleft_high))
            left_y_inds = np.where((win_y_low <=  nonzeroy) & (nonzeroy < win_y_high))
            good_left_inds = np.intersect1d(left_x_inds,left_y_inds,assume_unique=False)
            
            right_x_inds = np.where((win_xright_low <= nonzerox) & (nonzerox < win_xright_high))
            right_y_inds = np.where((win_y_low <=  nonzeroy) & (nonzeroy < win_y_high))
            good_right_inds = np.intersect1d(right_x_inds,right_y_inds,assume_unique=False)
            ###------------------------------------------------------------------------------------

            ### UDACITY SOLUTION: Identify the nonzero pixels in x and y within the window ###
            # good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            # (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            # good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            # (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            if debug:
                print()
                print(' left_x_inds  : ', left_x_inds[0].shape, ' left_y_indx  : ', left_y_inds[0].shape) 
                print(' good left inds size: ', good_left_inds.shape[0])
                # print(' X: ', nonzerox[good_left_inds]) ; print(' Y: ', nonzeroy[good_left_inds])
                print(' right_x_inds : ', right_x_inds[0].shape, ' right_y_indx : ', right_y_inds[0].shape)  
                print(' good right inds size: ', good_right_inds.shape[0])
                # print(' X: ', nonzerox[good_right_inds]); print(' Y: ', nonzeroy[good_right_inds])
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### If #pixels found > MINPIX pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position    ###
            if (self.MAXPIX > good_left_inds.shape[0] > self.MINPIX):
                left_msg  = ' Set leftx_current  :  {} ---> {} '.format( leftx_current,  int(nonzerox[good_left_inds].mean()))
                leftx_current = int(nonzerox[good_left_inds].mean())
            else:
                left_msg  = ' Keep leftx_current :  {} '.format(leftx_current)

            if (self.MAXPIX > good_right_inds.shape[0] > self.MINPIX ) :
                right_msg = ' Set rightx_current :  {} ---> {} '.format( rightx_current, int(nonzerox[good_right_inds].mean()))
                rightx_current = int(nonzerox[good_right_inds].mean())
            else:
                right_msg = ' Keep rightx_current:  {} '.format(rightx_current)
            
            if debug:
                print(left_msg)
                print(right_msg)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds  = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            print(' concatenate not working ')
            pass
        
        # Extract left and right line pixel positions
        self.LeftLane.allx  = nonzerox[left_lane_inds]
        self.LeftLane.ally  = nonzeroy[left_lane_inds] 
        self.RightLane.allx = nonzerox[right_lane_inds]
        self.RightLane.ally = nonzeroy[right_lane_inds]
        
        if debug:
            print()
            print(' leftx : ', self.LeftLane.allx.shape, ' lefty : ', self.LeftLane.ally.shape)
            print('   X:', self.LeftLane.allx[:15])
            print('   Y:', self.LeftLane.ally[:15])
            print('   X:', self.LeftLane.allx[-15:])
            print('   Y:', self.LeftLane.ally[-15:])
            print(' rightx : ', self.RightLane.allx.shape, ' righty : ', self.RightLane.ally.shape)
            print('   X:', self.RightLane.allx[:15])
            print('   Y:', self.RightLane.ally[:15])
            print('   X:', self.RightLane.allx[-15:])
            print('   Y:', self.RightLane.ally[-15:])

        self.processSlidingWin = True

        return out_img, histogram
'''

'''             
    def search_around_poly_v1(self, binary_warped, debug = False):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # MARGIN = 100

        # Grab activated pixels
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        # left_fit = self.LeftLane.current_fit
        # right_fit = self.RightLane.current_fit
        left_fit  = self.LeftLane.best_fit
        right_fit = self.RightLane.best_fit
        
        fitted_x_left     = (left_fit[0] * nonzeroy**2) + (left_fit[1] * nonzeroy) + (left_fit[2])
        fitted_x_right    = (right_fit[0]* nonzeroy**2) + (right_fit[1]* nonzeroy) + (right_fit[2])
            
        left_lane_inds  = ((nonzerox > fitted_x_left  - self.WINDOW_MARGIN) & (nonzerox < fitted_x_left  + self.WINDOW_MARGIN)).nonzero()
        right_lane_inds = ((nonzerox > fitted_x_right - self.WINDOW_MARGIN) & (nonzerox < fitted_x_right + self.WINDOW_MARGIN)).nonzero()
        
        if debug:
            print(' Search_around_poly() ')
            print(' fitted_x_left  : ', fitted_x_left.shape     , '  fitted_x_right : ', fitted_x_right.shape)
            print(' left_lane_inds : ', left_lane_inds[0].shape , left_lane_inds)
            print(' right_lane_inds: ', right_lane_inds[0].shape, right_lane_inds)
        
        # Extract left and right line pixel positions
        self.LeftLane.allx  = nonzerox[left_lane_inds]
        self.LeftLane.ally  = nonzeroy[left_lane_inds] 
        self.RightLane.allx = nonzerox[right_lane_inds]
        self.RightLane.ally = nonzeroy[right_lane_inds]
        
        return out_img, histogram
'''
        ################################################################################################
        ###----------------------------------------------------------------------------------------------
        ### Display undistorted color image & perpective transformed image -- WITHOUT RoI line display
        ###----------------------------------------------------------------------------------------------
        # if debug:
        #     print('imgUndist shape       :', imgUndist.shape, imgUndist.min(), imgUndist.max())
        #     print('imgWarped shape       :', imgWarped.shape, imgWarped.min(), imgWarped.max())
        #     display_two(imgUndist  , imgWarped, title1 = 'imgUndist',title2 = ' imgWarped', winttl = frameTitle)
        ###----------------------------------------------------------------------------------------------
        ### Display undistorted color image & perpective transformed image -- With RoI line display
        ###----------------------------------------------------------------------------------------------
        ### imgRoI   = draw_roi(imgUndist, RoI_vertices_list, thickness = 2)
        ### imgRoIWarped, _, _ = perspectiveTransform(imgRoI, src_points, dst_points, debug = False)
        ### print('imgRoI shape       :', imgRoI.shape, imgRoI.min(), imgRoI.max())
        ### print('imgRoIWarped shape       :', imgRoIWarped.shape, imgRoIWarped.min(), imgRoIWarped.max())
        ### display_two(imgRoI  , imgRoIWarped, title1 = 'imgRoI',title2 = ' imgRoIWarped', winttl = frameTitle)
        ###----------------------------------------------------------------------------------------------
        ### Display thresholded image befoire and after Perspective transform WITHOUT RoI line display
        ###----------------------------------------------------------------------------------------------
        # if debug:
        #     print('img Thrshld Warped shape:', imgThrshldWarped.shape, imgThrshldWarped.min(), imgThrshldWarped.max())
        #     display_two(imgThrshld, imgThrshldWarped, title1 = 'imgThrshld',title2 = 'imgThrshldWarped', winttl = frameTitle)
        ###----------------------------------------------------------------------------------------------
        ### Display thresholded image befoire and after Perspective transform WITH RoI line display
        ###----------------------------------------------------------------------------------------------
        ### imgThrshldRoI = draw_roi(imgThrshld, RoI_vertices_list, thickness = 1)
        ### imgThrshldRoIWarped, M, Minv = perspectiveTransform(imgThrshldRoI, src_points, dst_points, debug = False)
        ### print('imgThrshldRoI shape     :', imgThrshldRoI.shape, imgThrshldRoI.min(), imgThrshldRoI.max())
        ### print('img Thrshld Warped shape:', imgThrshldWarped.shape, imgThrshldWarped.min(), imgThrshldWarped.max())
        ### display_two(imgThrshldRoI, imgThrshldRoIWarped, title1 = 'imgThrshldRoI',title2 = 'imgThrshldRoIWarped', winttl = frameTitle)
        ###----------------------------------------------------------------------------------------------
        ### Display thresholded image without and with RoI line display
        ###----------------------------------------------------------------------------------------------
        ### display_two(imgThrshld, imgThrshldRoI, title1 = 'imgThrshld',title2 = 'imgThrshldRoI', winttl = frameTitle)
        ###----------------------------------------------------------------------------------------------
        ### Display MASKED color image with non RoI regions masked out -- With RoI line display
        ###----------------------------------------------------------------------------------------------
        ### imgMaskedDbg = region_of_interest(imgRoI, RoI_vertices)
        ### imgMaskedWarpedDbg, _, _ = perspectiveTransform(imgMaskedDbg, src_points, dst_points, debug = False)
        ### print('imgMaskedDebug shape    :', imgMaskedDbg.shape, imgMaskedDbg.min(), imgMaskedDbg.max())
        ### print('imgMaskedWarpedDbg shape:', imgMaskedWarpedDbg.shape, imgMaskedWarpedDbg.min(), imgMaskedWarpedDbg.max())
        ### display_two(imgMaskedDbg  , imgMaskedWarpedDbg, title1 = 'imgMaskedDebug',title2 = ' imgMaskedWarpedDebug', winttl = frameTitle)
        ###----------------------------------------------------------------------------------------------
        ### Display MASKED color image with non RoI regions masked out -- WITHOUT RoI line display
        ###----------------------------------------------------------------------------------------------
        ### imgMaskedDbg = region_of_interest(imgUndist, RoI_vertices)
        ### imgMaskedWarpedDbg, _, _ = perspectiveTransform(imgMaskedDbg, src_points, dst_points, debug = False)
        ### print('imgMaskedDebug shape    :', imgMaskedDbg.shape, imgMaskedDbg.min(), imgMaskedDbg.max())
        ### print('imgMaskedWarpedDbg shape:', imgMaskedWarpedDbg.shape, imgMaskedWarpedDbg.min(), imgMaskedWarpedDbg.max())
        ### display_two(imgMaskedDbg  , imgMaskedWarpedDbg, title1 = 'imgMaskedDebug',title2 = ' imgMaskedWarpedDebug', winttl = frameTitle)
        ###----------------------------------------------------------------------------------------------
        ###  Display Image warped before Thresholding and Warped AFTER Thresholding
        ###----------------------------------------------------------------------------------------------
        # if debug:
        #     imgWarpedThrshldList  = apply_thresholds(imgWarped, ksize=ksize, 
        #                                          x_thr = grad_x_thr, y_thr = grad_y_thr, 
        #                                        mag_thr = (50,255), dir_thr = (0,10), 
        #                                        sat_thr = (80, 255), debug = False)
        #     imgWarpedThrshld = imgWarpedThrshldList[-1]
        #     print(imgWarpedThrshld.shape)
        #     display_two( imgWarpedThrshld, imgThrshldWarped, 
        #                  title1 = 'Warped BEFORE Thresholding', 
        #                  title2 = 'Image warped AFTER Thresholding', winttl = frameTitle)
        ################################################################################################


