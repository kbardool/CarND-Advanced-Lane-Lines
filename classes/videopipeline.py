import os
import sys
if '..' not in sys.path:
    print("pipeline.py: appending '..' to sys.path")
    sys.path.append('..')
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pprint 
import copy 
import winsound 
from collections import deque, defaultdict
from classes.line import Line
from classes.plotdisplay import PlotDisplay
from common.utils import (sliding_window_detection  , polynomial_proximity_detection, 
                          offCenterMsg      , curvatureMsg      , colorLanePixels   , displayPolynomial      , displayRoILines,  
                          displayDetectedRegion   , displayText , displayGuidelines , displayPolySearchRegion, 
                          display_one, display_two, display_multi )
from common.sobel import  apply_thresholds,  apply_perspective_transform, perspectiveTransform, erodeDilateImage
                                                                                
pp = pprint.PrettyPrinter(indent=2, width=100)
print(' Loading pipeline.py - cwd:', os.getcwd())

class VideoPipeline(object):
    NAME                     = 'ALFConfig'
    
    def __init__(self, cameraConfig, **kwargs):

        self.camera                     = cameraConfig
        self.height                     = self.camera.height
        self.width                      = self.camera.width
        self.camera_x                   = self.camera.width //2
        self.camera_y                   = self.camera.height
  
        self.debug                      = kwargs.get('debug'                , False)
        self.debug2                     = kwargs.get('debug2'               , False)
        self.debug3                     = kwargs.get('debug3'               , False)
        self.displayResults             = kwargs.get('displayResults'       , False)
        self.displayFittingInfo         = kwargs.get('displayFittingInfo'   , False)
        self.displayRealignment         = kwargs.get('displayRealignment'   , False)
        self.overlayBeta                = kwargs.get('overlayBeta'          , 0.7)
        self.ERODE_DILATE               = kwargs.get('erode_dilate'         , False)        
  
        self.mode                       = kwargs.get('mode'                 ,    1)
        self.POLY_DEGREE                = kwargs.get('poly_degree'          ,    2)
        self.MIN_POLY_DEGREE            = kwargs.get('min_poly_degree'      ,    2)
        self.MIN_X_SPREAD               = kwargs.get('min_x_spread', 90)
        self.MIN_Y_SPREAD               = kwargs.get('min_y_spread', 350)


        self.HISTORY                    = kwargs.get('history'              ,    8)
        self.COMPUTE_HISTORY            = kwargs.get('compute_history'      ,    2)
        
        self.NWINDOWS                   = kwargs.get('nwindows'             ,   30)
        self.HISTOGRAM_WIDTH_RANGE      = kwargs.get('hist_width_range'     ,  600)
        self.HISTOGRAM_DEPTH_RANGE      = kwargs.get('hist_depth_range'     ,  2 * self.height // 3)
        self.WINDOW_SRCH_MRGN           = kwargs.get('window_search_margin' ,   55)
        self.INIT_WINDOW_SRCH_MRGN      = kwargs.get('init_window_search_margin' ,  self.WINDOW_SRCH_MRGN)
        self.MINPIX                     = kwargs.get('minpix'               ,   90)
        self.MAXPIX                     = kwargs.get('maxpix'               , 8000)
 
        self.POLY_SRCH_MRGN             = kwargs.get('poly_search_margin'   ,   45)
        
        self.IMAGE_RATIO_HIGH_THRESHOLD = kwargs.get('image_ratio_high_threshold',   40)
        self.IMAGE_RATIO_LOW_THRESHOLD  = kwargs.get('image_ratio_low_threshold' ,    2)

        self.LANE_COUNT_THRESHOLD       = kwargs.get('lane_count_threshold'      , 4500)
        self.LANE_RATIO_LOW_THRESHOLD   = kwargs.get('lane_ratio_low_threshold'  ,    2)
        self.LANE_RATIO_HIGH_THRESHOLD  = kwargs.get('lane_ratio_high_threshold' ,   60)
        
        self.RSE_THRESHOLD              = kwargs.get('rse_threshold'           ,  80)

        self.PARALLEL_LINES_MARGIN      = kwargs.get('parallel_lines_margin'   ,  70)
        self.YELLOW_DETECTION_LIMIT     = kwargs.get('yello_limit'             ,  25)
        self.RED_DETECTION_LIMIT        = kwargs.get('red_limit'               ,  50)
        self.OFF_CENTER_ROI_THRESHOLD   = kwargs.get('off_center_roi_threshold',  60)
        self.CURRENT_OFFCTR_ROI_THR     = np.copy(self.OFF_CENTER_ROI_THRESHOLD)

        self.HISTOGRAM_SEARCH_RANGE   = (self.camera_x - self.HISTOGRAM_WIDTH_RANGE, self.camera_x + self.HISTOGRAM_WIDTH_RANGE)

        ## Thresholding Parameters 
        self.HIGH_RGB_THRESHOLD       = kwargs.get('high_rgb_threshold'   ,  180)   # 220
        self.MED_RGB_THRESHOLD        = kwargs.get('med_rgb_threshold'    ,  180)   # 175   ## chgd from 110 2-26-20
        self.LOW_RGB_THRESHOLD        = kwargs.get('low_rgb_threshold'    ,  100)   # 175   ## chgd from 110 2-26-20
        self.VLOW_RGB_THRESHOLD       = kwargs.get('vlow_rgb_threshold'   ,   35)   # 175   ## chgd from 110 2-26-20

        self.XHIGH_SAT_THRESHOLD      = kwargs.get('xhigh_sat_threshold'  ,  120)   # 150
        self.HIGH_SAT_THRESHOLD       = kwargs.get('high_sat_threshold'   ,   65)   # 150
        self.LOW_SAT_THRESHOLD        = kwargs.get('low_sat_threshold'    ,   20)   #  20   ## chgd from 110 2-26-20

        self.XHIGH_THRESHOLDING       = kwargs.get('xhigh_thresholding'   , 'cmb_mag_x')
        self.HIGH_THRESHOLDING        = kwargs.get('high_thresholding'    , 'cmb_mag_x')
        self.NORMAL_THRESHOLDING      = kwargs.get('med_thresholding'     , 'cmb_rgb_lvl_sat')
        self.LOW_THRESHOLDING         = kwargs.get('low_thresholding'     , 'cmb_mag_xy')
        self.VLOW_THRESHOLDING        = kwargs.get('vlow_thresholding'    , 'cmb_mag_xy')
        self.HISAT_THRESHOLDING       = kwargs.get('hisat_thresholding'   , 'cmb_mag_x')
        self.LOWSAT_THRESHOLDING      = kwargs.get('lowsat_thresholding'  , 'cmb_hue_x')
        
        # self.DARK_THRESHOLDING      = 'cmb_mag_x'
        # self.lowsat_thresholding    = 'cmb_rgb_lvl_sat_mag'
        # self.NORMAL_THRESHOLDING    = 'cmb_rgb_lvl_sat_mag_x'

        ## set threshold limits for various conditions
        self.initialize_thresholding_parameters()

        self.slidingWindowBootstrap   = True
        self.firstFrame               = True
        self.RoIAdjustment            = False
        self.validLaneDetections      = False
        self.imgThrshldHistory        = []
        self.imgCondHistory           = []
        self.imgAcceptHistory         = []
        self.imgAdjustHistory         = [] 
        self.diffsSrcDynPoints        = []
        self.offctr_history           = []
        self.imgPixelRatio            = []
        self.src_points_history       = [] 
        self.HLS_key                  = ['Hue', 'Lvl', 'Sat']
        self.RGB_key                  = ['Red', 'Grn', 'Blu']
        self.imgUndistStats           = self.initImageInfoDict()
        self.imgWarpedStats           = self.initImageInfoDict()

        self.ttlFullReject      = 0
        self.ttlSkipFrameDetect = 0 
        self.ttlRejectedFrames  = 0 
        self.ttlAcceptedFrames  = 0 
        self.ttlRejectedFramesSinceAccepted = 0 
        self.ttlAcceptedFramesSinceRejected = 0 

        ## Parameters for perspective transformation source/destination points 

        self.y_src_top                = kwargs.get('y_src_top'      ,  480)          ## 460 -> 465 y_src_bot - 255
        self.y_src_bot                = kwargs.get('y_src_bot'      ,  self.height)          ## image.shape[0] - 20
        self.RoI_x_adj                = kwargs.get('RoI_x_adj'      ,   25)
        self.lane_theta               = kwargs.get('lane_theta'     ,   40)    ## Lane Angle 
        self.x_bot_disp               = kwargs.get('bot_x_disp'     ,  375)
        
        self.x_dst_left               = kwargs.get('x_dst_left'     ,   300)
        self.x_dst_right              = kwargs.get('x_dst_right'    ,  1000)
        self.y_dst_top                = kwargs.get('y_dst_top'      ,     0)
        self.y_dst_bot                = kwargs.get('y_dst_bot'      ,  self.height - 1)    

        ## Parameters indicating extent of detected region to be displayed on final image
        self.displayRegionTop         = kwargs.get('displayRegionTop'  ,  self.y_src_top)
        self.displayRegionBot         = kwargs.get('displayRegionBot'  ,  self.y_src_bot)
        
        print(' y_src_bot: ', self.y_src_bot, ' displayRegionBot : ', self.displayRegionBot)
        
        self.src_points_list, self.src_points = self.build_source_RoI_region()
        self.dst_points_list, self.dst_points = self.build_dest_RoI_region()
        self.prev_src_points_list = copy.copy(self.src_points_list)

        ## Destination points for Perspective Transform                           
        self.curvature_y_eval     = self.y_src_bot
        self.offCenter_y_eval     = self.y_src_bot

        self.np_format = {
                'float' : lambda x: "%7.2f" % x,
                'int'   : lambda x: "%5d" % x
        }
        np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =500, formatter = self.np_format)
    
        self.LeftLane = Line(name =  'Left', history = self.HISTORY, compute_history = self.COMPUTE_HISTORY,
                             poly_degree = self.POLY_DEGREE, min_poly_degree = self.MIN_POLY_DEGREE,
                             min_x_spread = self.MIN_X_SPREAD, min_y_spread = self.MIN_Y_SPREAD,   
                             height = self.height,  y_src_top = self.y_src_top, y_src_bot = self.y_src_bot, 
                             rse_threshold = self.RSE_THRESHOLD)
        self.RightLane= Line(name = 'Right', history = self.HISTORY,  compute_history = self.COMPUTE_HISTORY,
                             poly_degree = self.POLY_DEGREE, min_poly_degree = self.MIN_POLY_DEGREE,
                             min_x_spread = self.MIN_X_SPREAD, min_y_spread = self.MIN_Y_SPREAD,                             
                             height = self.height, y_src_top = self.y_src_top, y_src_bot = self.y_src_bot, 
                             rse_threshold = self.RSE_THRESHOLD)
        print(' Pipeline initialization complete...')                                      



    def initImageInfoDict(self):
        
        plots_dict = {}
        plots_dict.setdefault('RGB', [])
        plots_dict.setdefault('HLS', [])
        for key1 in self.RGB_key + self.HLS_key :  ##  ['Hue', 'Lvl', 'Sat', 'Red', 'Grn', 'Blu', 'RGB']:
            plots_dict.setdefault(key1, [])

        return plots_dict

    def saveImageStats(self, image, imageDict):
        
        imgHLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        imageDict['RGB'].append(np.round(image.mean(),0))
        imageDict['HLS'].append(np.round(imgHLS.mean(),0))

        img_RGB_Avgs = np.round(image.mean(axis=(0,1)),0)
        img_HLS_Avgs = np.round(imgHLS.mean(axis=(0,1)),0)

        for i,key in enumerate(self.RGB_key):
            imageDict[key].append(img_RGB_Avgs[i])
        for i,key in enumerate(self.HLS_key):
            imageDict[key].append(img_HLS_Avgs[i])


    def process_one_frame(self, **kwargs):
        
        self.debug  = kwargs.get('debug'  , True)
        self.debug2 = kwargs.get('debug2' , True)
        self.debug3 = kwargs.get('debug3' , False)
        read_next   = kwargs.get('read_next', True)
        size        = kwargs.get('size', (15,7))
        show        = kwargs.get('show', True)
        # display   = kwargs.get('display', True)
        self.displayResults     = kwargs.get('displayResults'    , self.displayResults    )
        self.displayFittingInfo = kwargs.get('displayFittingInfo', self.displayFittingInfo)
        self.displayRealignment = kwargs.get('displayRealignment', self.displayRealignment)

        # print(kwargs)
        # print(f' displayFittingInfo: {self.displayFittingInfo}    displayRealignment:{self.displayRealignment}     displayResults:{self.displayResults}')
        
        if read_next:
            rc1= self.inVideo.getNextFrame()  
        else:
            rc1 = True
            
        if rc1:
            outputImage, disp = self(displayResults = self.displayResults,
                                     displayFittingInfo = self.displayFittingInfo, 
                                     displayRealignment = self.displayRealignment, 
                                     debug = self.debug, debug2 = self.debug2, debug3 = self.debug3)
            self.outVideo.saveFrameToVideo(outputImage, debug = False)        
            
            # _ = display_one(outputImage, size=size, title = self.frameTitle)
            
            winsound.MessageBeep(type=winsound.MB_ICONHAND)

        return (outputImage, disp)


    def process_frame_range(self, toFrame, **kwargs):
        
        self.debug       = kwargs.get('debug'  , False)
        self.debug2      = kwargs.get('debug2' , False)
        self.debug3      = kwargs.get('debug3' , False)
        display          = kwargs.get('display', False)
        disp_interval    = kwargs.get('disp_interval', 50)
        size             = kwargs.get('size', (15,5))
        show             = kwargs.get('show', True)
        self.displayResults     = kwargs.get('displayResults'    , self.displayResults    )
        self.displayFittingInfo = kwargs.get('displayFittingInfo', self.displayFittingInfo)
        self.displayRealignment = kwargs.get('displayRealignment', self.displayRealignment)

        print(' displayFittingInfo: ', self.displayFittingInfo, ' displayRealignment:', self.displayRealignment, '  displayResults: ', self.displayResults)

        print('From : ', self.inVideo.currFrameNum, ' To:', toFrame)
        
        rc1     = True
        while self.inVideo.currFrameNum < toFrame  and rc1:    
            rc1 =  self.inVideo.getNextFrame()
            if rc1:
                output, disp = self(displayResults = self.displayResults,
                                    displayFittingInfo = self.displayFittingInfo, 
                                    displayRealignment = self.displayRealignment,
                                    debug = self.debug, debug2 = self.debug2, debug3 = self.debug3)
                self.outVideo.saveFrameToVideo(output, debug = self.debug)        

            if show and (self.inVideo.currFrameNum % disp_interval == 0):      ##  or (110 <=Pipeline.inVideo.currFrameNum <=160) :
                
                display_two(self.prevBestFit, self.imgLanePxls, size = (15,5), 
                            title1 = 'Prev best fit (Cyan: Prev fit, Yellow: New proposal)' , 
                            title2 = 'ImgLanePxls (Cyan: Prev fit, Yellow: New proposal, Fuschia: New Best Fit)' )
                display_one(output, size= size, title = self.inVideo.frameTitle)        
                
        print('Finshed - Curr frame number :', self.inVideo.currFrameNum)
        return


    def __call__(self, **kwargs ):
        '''
        '''
        self.debug                    = kwargs.get('debug' , False)
        self.debug2                   = kwargs.get('debug2', False)
        self.debug3                   = kwargs.get('debug3', False)
        self.debug4                   = kwargs.get('debug4', False)
        self.displayResults           = kwargs.get('displayResults'    , self.displayResults    )
        self.displayFittingInfo       = kwargs.get('displayFittingInfo', self.displayFittingInfo)
        self.displayRealignment       = kwargs.get('displayRealignment', self.displayRealignment)
        self.exit                     = kwargs.get('exit'  , 0)             
        self.mode                     = kwargs.get('mode'  , self.mode)
        self.slidingWindowBootstrap   = kwargs.get('slidingWindowBootstrap'  , self.slidingWindowBootstrap) 
        self.image                    = self.inVideo.image   
        self.frameTitle               = self.inVideo.frameTitle
        self.resultExtraInfo         = None

        ###----------------------------------------------------------------------------------------------
        ### PIPELINE 
        ###----------------------------------------------------------------------------------------------
        self.imgUndist = self.camera.undistortImage(self.image)

        self.saveImageStats(self.imgUndist, self.imgUndistStats)

        self.src_points_history.append(self.src_points)

        self.imgWarped, self.M , self.Minv = perspectiveTransform(self.imgUndist, self.src_points, self.dst_points, debug = self.debug4)
        
        self.saveImageStats(self.imgWarped, self.imgWarpedStats)
        
        self.imgRoI             = displayRoILines(self.imgUndist, self.src_points_list, thickness = 2)
        self.imgRoIWarped, _, _ = perspectiveTransform(self.imgRoI   , self.src_points     , self.dst_points)
        self.imgRoIWarped       = displayRoILines(self.imgRoIWarped  , self.dst_points_list, thickness = 2, color = 'yellow')

        ###----------------------------------------------------------------------------------------------
        ### Select image to process based on MODE parameter, and select thrsholding parameters
        ###----------------------------------------------------------------------------------------------
        self.set_thresholding_parms()

        ###----------------------------------------------------------------------------------------------
        ### Debug Info
        ###----------------------------------------------------------------------------------------------        

        if self.debug:
            self.debugInfo_ImageInfo()
            self.debugInfo_ImageSummaryInfo()
            self.debugInfo_srcPointsRoI(title= 'Perspective Tx. source points')

        ###----------------------------------------------------------------------------------------------
        ### Apply thresholding and Warping of thresholded images 
        ###----------------------------------------------------------------------------------------------
        if self.mode == 1:
            self.image_to_threshold = self.imgUndist
        else:
            self.image_to_threshold = self.imgWarped

        outputs = apply_thresholds(self.image_to_threshold, self.thresholdParms)

        if self.mode == 1:
            warped_outputs = apply_perspective_transform(outputs, self.thresholdStrs, self.src_points, self.dst_points, 
                                                        size = (15,5), debug = self.debug)
            self.working_image = warped_outputs[self.thresholdMethod]
            self.imgThrshld = outputs[self.thresholdMethod]
        else:
            self.working_image = outputs[self.thresholdMethod]
            self.imgThrshld = outputs[self.thresholdMethod]
        
        # display_one(self.imgThrshld, size=(15,7), title = 'imgThrshld')
        # display_two(self.imgThrshld, self.working_image, title1 = 'imgThrshld', title2 = 'working_image')

        # if self.exit == 1:
            # return self.imgThrshld, None

        ###----------------------------------------------------------------------------------------------
        ##  if ERODE_DILATE flag is True, erode/dilate thresholded image
        ###----------------------------------------------------------------------------------------------
        # if self.+mode == 1:  ### Warped AFTER thresholding
            # self.post_threshold, _, Minv = perspectiveTransform(self.imgThrshld, self.src_points, self.dst_points, debug = self.debug4)
        # else:               ### Warped BEFORE thresholding
            # self.post_threshold = self.imgThrshld

        # if self.ERODE_DILATE:
            # self.working_image = erodeDilateImage(self.post_threshold , ksize = 3, iters = 3)
        # else:
            # self.working_image = self.post_threshold
        
        # self.working_image = self.post_threshold


        ###----------------------------------------------------------------------------------------------
        ## INTERMEDIATE DEBUG DISPLAYS
        ###----------------------------------------------------------------------------------------------
        # if debug  and displayResults:   ##  and self.mode == 2:
        if self.debug:
            self.debugInfo_ThresholdedImage()
        
        ###----------------------------------------------------------------------------------------------
        ### Find lane pixels 
        ###----------------------------------------------------------------------------------------------
        if self.slidingWindowBootstrap:
            window_search_margin = self.INIT_WINDOW_SRCH_MRGN if self.firstFrame else self.WINDOW_SRCH_MRGN
            reset_search_base    = (self.firstFrame  or self.imgAcceptHistory[-1] < -10) 
            if self.RoIAdjustment  :
                reset_search_base = False
            self.out_img, self.histogram, self.detStats = sliding_window_detection(self.working_image, 
                                                                           self.LeftLane, self.RightLane, 
                                                                           nwindows        = self.NWINDOWS, 
                                                                           histWidthRange  = self.HISTOGRAM_WIDTH_RANGE, 
                                                                           histDepthRange  = self.HISTOGRAM_DEPTH_RANGE, 
                                                                           search_margin   = window_search_margin, 
                                                                           reset_search_base = reset_search_base,
                                                                           debug = self.debug,
                                                                           debug2 = self.debug2) 

        else:    
            self.out_img, self.histogram, self.detStats = polynomial_proximity_detection(self.working_image, 
                                                                             self.LeftLane, self.RightLane, 
                                                                             search_margin   = self.POLY_SRCH_MRGN, 
                                                                             debug = self.debug)
        if self.debug:
            self.debugInfo_LaneDetInfo()
    
        self.assess_lane_detections()
        
        # if self.exit == 2:
            # return self.out_img, None        
        
        ###----------------------------------------------------------------------------------------------
        ### Fit polynomial on found lane pixels 
        ###----------------------------------------------------------------------------------------------
        for Lane in [self.LeftLane, self.RightLane]:
            Lane.fit_polynomial(debug  = self.debug)

        self.assess_fitted_polynomials()
        
        if self.debug:
            self.debugInfo_DetectedLanes(display=0, size = (15,5))

        if self.displayFittingInfo:
            self.debugInfo_displayFittingInfo()            

        ###----------------------------------------------------------------------------------------------
        ### Build output image frame  
        ###----------------------------------------------------------------------------------------------
        self.build_result_image() 
        
        ###----------------------------------------------------------------------------------------------
        ### Determine if an adjustment of the Perspective transformation window is necessary and if so, 
        ### adjust the SRC_POINTS_LIST and/or DST_POINTS_LIST accordingly 
        ###----------------------------------------------------------------------------------------------
        self.adjust_RoI_window()

        ###----------------------------------------------------------------------------------------------
        ### All done - build display results if requested 
        ###----------------------------------------------------------------------------------------------        
        if self.displayResults:
            self.build_display_results()

        if self.firstFrame :
            self.firstFrame = False

        return self.resultImage, self.resultExtraInfo
            

    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def assess_lane_detections(self):

        imgPixelRatio, self.NztoSrchNzRatio, self.NztoImageNzRatio, ttlImageNZPixels, ttlLaneNZPixels = self.detStats
        self.imgPixelRatio.append(imgPixelRatio)

        lower_nz_pxl_cnt = round(np.sum(self.working_image[480:,:]) * 100/(self.height*self.width//3),2)

        if self.debug:
            print()
            print('assess_lane_detections()')
            print('-'*40)
            print(' Lower image non_zero pixel ratio:  %{:8.2f}'.format(lower_nz_pxl_cnt))
            print(' (Image NZ pixels to Total Pixels in image)          imgPixelRatio : %{:8.2f} \n'\
                  ' (Detected NZ Pixels to All pixels in Search Region) NZtoSrNZRatio : %{:8.2f} \n' \
                  ' (Detected NZ Pixels to All NZ Pixels in image)      NztoTtlNzRatio: %{:8.2f}'.format(
                      imgPixelRatio , self.NztoSrchNzRatio , self.NztoImageNzRatio ))
            print()        

        msgs = []
        image_conditions = []
        
        ##------------------------------------------------------------------------------------------
        ##  Frame / Lane detection Quality checks 
        ##------------------------------------------------------------------------------------------
        for Lane in [self.LeftLane, self.RightLane]:
            
            if (Lane.pixelCount[-1] < self.LANE_COUNT_THRESHOLD): 
                image_conditions.append(10)        
                msgs.append(' ***  (10) {:5s} Lane pixel count under threshold - Pxl Count: {:7.0f} < Count Threshold: ({:4d}) '.format(
                        Lane.name, Lane.pixelCount[-1], self.LANE_COUNT_THRESHOLD))
                Lane.goodLaneDetection = False

            elif (Lane.pixelRatio[-1]  < self.LANE_RATIO_LOW_THRESHOLD): 
                image_conditions.append(11)
                msgs.append(' ***  (11) {:5s} Lane pixel ratio under threshold - Pxl Ratio: {:7.3f} < Ratio Threshold: ({:7.3f}) '\
                      ' Pxl Count: {:7.0f} - Count Threshold: ({:4d})'.format(Lane.name, 
                       Lane.pixelRatio[-1], self.LANE_RATIO_LOW_THRESHOLD, Lane.pixelCount[-1], self.LANE_COUNT_THRESHOLD))
                Lane.goodLaneDetection = False

            elif (Lane.pixelRatio[-1]  > self.LANE_RATIO_HIGH_THRESHOLD) and \
                 (self.NztoImageNzRatio < 30): 
                image_conditions.append(12)
                msgs.append(' ***  (12) {:5s} Lane pxl ratio > threshold - Pxl Ratio: {:7.3f} > Ratio Threshold: ({:7.3f}) '\
                      ' Det Nz to Ttl Nz Ratio: ({:7.3f})'.format(Lane.name, 
                       Lane.pixelRatio[-1], self.LANE_RATIO_HIGH_THRESHOLD, self.NztoImageNzRatio))
                Lane.goodLaneDetection = False
         
            else:
                Lane.goodLaneDetection = True


        ##------------------------------------------------------------------------------------------
        ##  Frame Level Quality checks 
        ##------------------------------------------------------------------------------------------
        self.frameGoodQuality = True
        self.bothLanesPixelRatio = self.LeftLane.pixelRatio[-1] + self.RightLane.pixelRatio[-1]
        
        if self.imgPixelRatio[-1] > self.IMAGE_RATIO_HIGH_THRESHOLD:     ##    self.IMAGE_RATIO_HIGH_THRESHOLD:
            image_conditions.append(20)
            msgs.append(' ***  (20) imgPixelRatio:  ratio of non-zero pixels in image {} > image ratio HIGH threshold {}'.
                        format(self.imgPixelRatio[-1],  self.IMAGE_RATIO_HIGH_THRESHOLD))
            self.frameGoodQuality = False

        if self.imgPixelRatio[-1] < self.IMAGE_RATIO_LOW_THRESHOLD:
            image_conditions.append(21)
            msgs.append(' ***  (21) imgPixelRatio:  ratio of non-zero pixels in image {} < image ratio LOW threshold {}'.
                        format(self.imgPixelRatio[-1],  self.IMAGE_RATIO_LOW_THRESHOLD))

        if self.bothLanesPixelRatio < self.IMAGE_RATIO_LOW_THRESHOLD:
            image_conditions.append(30)
            msgs.append(' ***  (30) bothLanesPixelRatio:  Left+Right non-zero pixel ratio {} < image ratio LOW threshold {}.'.
                        format(self.bothLanesPixelRatio, self.IMAGE_RATIO_LOW_THRESHOLD))

        # if self.bothLanesPixelRatio > self.IMAGE_RATIO_HIGH_THRESHOLD:
            # image_conditions.append(31)
            # msgs.append(' ***  (31) bothLanesPixelRatio:  Left+Right non-zero pixel ratio {} > image ratio HIGH threshold {}.'.
                        # format(self.bothLanesPixelRatio, self.IMAGE_RATIO_LOW_THRESHOLD))

        if (lower_nz_pxl_cnt > 45 ):
            image_conditions.append(40)
            msgs.append(' ***  (31) Warped image lower 1/3 non-zero pixel count {}  > 45  '.format(lower_nz_pxl_cnt))
            self.frameGoodQuality = False

        if (self.imgWarpedStats['RGB'][-1]> self.HIGH_RGB_THRESHOLD) and (self.imgWarpedStats['Sat'][-1] > self.XHIGH_SAT_THRESHOLD):
            image_conditions.append(40)
            msgs.append(' ***  (40) Warped Image High Mean RGB {} / Mean SAT {}  '.
                    format(self.imgWarpedStats['RGB'][-1], self.imgWarpedStats['Sat'][-1]))
            self.frameGoodQuality = False

        self.goodLaneDetections = (self.LeftLane.goodLaneDetection and  self.RightLane.goodLaneDetection) 
        self.imgCondHistory.append(image_conditions)
        
        if self.debug:
            print(' Image conditions:  ', image_conditions)
            for msg in msgs:
                print(msg)

            print()
            print(' left Pxl Count: {:7.0f}  or  right Pxl Count: {:7.0f} - LANE_COUNT_THRESHOLD  : {:7.0f} '.
                    format(self.LeftLane.pixelCount[-1], self.RightLane.pixelCount[-1], self.LANE_COUNT_THRESHOLD))
            print(' left Pxl Ratio: {:7.2f}  or  right Pxl Ratio: {:7.2f} - LANE RATIO LOW THRSHLD: {:7.2f}    HIGH THRSHLD {:7.2f}'.
                    format(self.LeftLane.pixelRatio[-1], self.RightLane.pixelRatio[-1], 
                    self.LANE_RATIO_LOW_THRESHOLD, self.LANE_RATIO_HIGH_THRESHOLD))
            print(' Image NZ pixel ratio (imgPixelRatio)        : {:7.2f} - IMG  RATIO LOW THRSHLD: {:7.2f}    HIGH THRSHLD {:7.2f}'.
                    format(self.imgPixelRatio[-1], self.IMAGE_RATIO_LOW_THRESHOLD, self.IMAGE_RATIO_HIGH_THRESHOLD))
            # print('  Left+Right    : %{:7.2f}    imgPixelRatio: %{:7.2f} '.
                    # format(self.bothLanesPixelRatio, self.imgPixelRatio[-1] ))
            print(' L+R NZ pixel ratio  (bothLanesPixelRatio)   : {:7.2f} - IMG  RATIO LOW THRSHLD: {:7.2f}    HIGH THRSHLD {:7.2f}'.
                    format(self.bothLanesPixelRatio, self.IMAGE_RATIO_LOW_THRESHOLD, self.IMAGE_RATIO_HIGH_THRESHOLD))
            print(' imgWarped stats  RGB:  {:7.2f}   SAT: {:7.2f}    HIGH_RGB_THRSHLD: {:7.2f} '\
                  '  HIGH_SAT_THRSHLD {:7.2f}    EXTRA HIGH_SAT_THRSHLD {:7.2f}'.
                    format(self.imgWarpedStats['RGB'][-1], self.imgWarpedStats['Sat'][-1], 
                    self.HIGH_RGB_THRESHOLD, self.HIGH_SAT_THRESHOLD, self.XHIGH_SAT_THRESHOLD))
            print()
            print(' Lane Detections Results -    Left: {}    Right: {}    goodLaneDetections: {}    frameGoodQuality: {}'.format(
                str(self.LeftLane.goodLaneDetection).upper(), str(self.RightLane.goodLaneDetection).upper(), 
                str(self.goodLaneDetections).upper()        , str(self.frameGoodQuality).upper() ))


    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def assess_fitted_polynomials(self):

        if self.debug:
            print()
            print('assess_fitted_polynomials()')
            print('-'*40)

        ### Individual lane assessments

        for Lane in [self.LeftLane, self.RightLane]:
            

            if (self.slidingWindowBootstrap and self.RoIAdjustment):
                ## Realignment of the perspective transformation window will reuslt in a 
                ## High RSE Error. We will allow this error rate when it is a result of a 
                ## RoI realignment. Other wise proceed nornally.                
                Lane.acceptPolynomial = True
                Lane.reset_best_fit(debug = self.debug)
                msg2 = '    {:5s} lane fitted polynomial - RoIAdjustment performed - Polynomial fit will be accepted \n'.format(Lane.name)

            elif (Lane.goodLaneDetection and self.frameGoodQuality):
                Lane.acceptPolynomial = True
                # Lane.reset_best_fit(debug = self.debug)
                msg2 = '    {:5s} lane fitted polynomial - acceptPolynomial: {}   (GoodLaneDetection: {} & frameGoodQuality: {})'.format(
                    Lane.name, Lane.acceptPolynomial, Lane.goodLaneDetection, self.frameGoodQuality)
            
            # elif not (Lane.goodLaneDetection and self.frameGoodQuality):
            #     Lane.acceptPolynomial = False
            #     msg2 = '    {:5s} lane fitted polynomial - acceptPolynomial: {}   (GoodLaneDetection: {} & frameGoodQuality: {})'.format(
            #         Lane.name, Lane.acceptPolynomial, Lane.goodLaneDetection, self.frameGoodQuality)
            
            elif Lane.curve_spread_x > (2 * Lane.pixel_spread_x):
                Lane.acceptPolynomial = False            
                msg2 = '    {:5s} lane fitted polynomial x spread {}  > 2*PixelSpread {} '.format(
                    Lane.name, Lane.curve_spread_x, (2 * Lane.pixel_spread_x))

            elif not (Lane.goodLaneDetection):
                Lane.acceptPolynomial = False            
                msg2 = '    {:5s} lane fitted polynomial - acceptPolynomial: {}   (GoodLaneDetection: {} & frameGoodQuality: {})'.format(
                    Lane.name, Lane.acceptPolynomial, Lane.goodLaneDetection, self.frameGoodQuality)
            else:
                Lane.acceptPolynomial =  True if (Lane.RSE < Lane.RSE_THRESHOLD) else False
                msg2 = '    {:5s} lane fitted polynomial - acceptPolynomial:  {}'.format(Lane.name, Lane.acceptPolynomial)

            if self.debug :
                print(msg2)        



        ### Joint Lane assessments

        if (self.LeftLane.acceptPolynomial ^ self.RightLane.acceptPolynomial) and (self.goodLaneDetections):
            self.compareLanes()

        for Lane in [self.LeftLane, self.RightLane]:
            if Lane.acceptPolynomial:
                Lane.acceptFittedPolynomial(debug = self.debug, debug2 = self.debug2)
            else:
                Lane.rejectFittedPolynomial(debug = self.debug, debug2 = self.debug2)


        ### Frame level actions that need to be taken based on acceptance or rejection of polynomials 

        self.acceptPolynomials = self.LeftLane.acceptPolynomial and self.RightLane.acceptPolynomial and self.frameGoodQuality
        fullReject    = not (self.LeftLane.acceptPolynomial or self.RightLane.acceptPolynomial or self.frameGoodQuality)
        # red_status    = not ((self.LeftLane.acceptPolynomial ^ self.RightLane.acceptPolynomial) ^ self.frameGoodQuality)
        # yellow_status = not red_status


        if self.acceptPolynomials:   ## everything good
            self.ttlAcceptedFrames += 1
            self.ttlRejectedFramesSinceAccepted = 0
            self.ttlAcceptedFramesSinceRejected += 1
            self.validLaneDetections     = True
            self.polyRegionColor1        = 'green'
            self.slidingWindowBootstrap  = False
            acceptCode = 0
        
        elif fullReject:            ## everything bad
            self.ttlFullReject += 1            
            self.ttlRejectedFramesSinceAccepted = 0
            self.ttlAcceptedFramesSinceRejected = 0
            self.slidingWindowBootstrap  = False
            self.validLaneDetections     = False
            self.polyRegionColor1        = 'lightgray'       
            acceptCode = -40
        
        else:     
            self.ttlRejectedFrames += 1
            self.ttlAcceptedFramesSinceRejected = 0 
            self.ttlRejectedFramesSinceAccepted += 1
            self.validLaneDetections  = True

            # self.slidingWindowBootstrap  = True if self.frameGoodQuality else False
            #  doesnt work well in YELLOW conditions.

            if self.ttlRejectedFramesSinceAccepted < self.YELLOW_DETECTION_LIMIT:
                self.slidingWindowBootstrap  = False
                self.polyRegionColor1       = 'yellow' 
                acceptCode = -10
            else:
                # 
                self.slidingWindowBootstrap  = True if self.frameGoodQuality else False
                self.polyRegionColor1       = 'red'      

                if self.ttlRejectedFramesSinceAccepted < self.RED_DETECTION_LIMIT:
                    acceptCode = -20
                else:
                    # self.polyRegionColor1 = 'lightgray'       
                    acceptCode = -30

        self.imgAcceptHistory.append(acceptCode)
  
        ### Display debug info
        if self.debug: 
            print()
            for lane in [self.LeftLane, self.RightLane]:
                if lane.acceptPolynomial:
                    print('=> {:5s} Lane ACCEPT polynomial - Accepted frames Since Last Rejected: {:4d}'.format(
                        lane.name, lane.ttlAcceptedFramesSinceRejected))
                else:
                    print('=> {:5s} Lane REJECT polynomial - Rejected frames Since Last Detected: {:4d}'.format(
                        lane.name, lane.ttlRejectedFramesSinceDetected))
            print()
            print('=> acceptPolynomials: {}    frameGoodQuality: ({})'.format(
                    str(self.acceptPolynomials).upper(),  str(self.frameGoodQuality).upper()  )) 
            print('   slidingWindowBootstrap: {}     validLaneDetections: {}    acceptCode: {}    displayColor: {}   '.format(
                    self.slidingWindowBootstrap, self.validLaneDetections, acceptCode, self.polyRegionColor1 ))
            print('   Total Accepted sinceLast Rejected: {:3d}   Rejected since Last Accepted: {:3d} \n'.format(
                    self.ttlAcceptedFramesSinceRejected, self.ttlRejectedFramesSinceAccepted  ))
            self.debugInfo_DetectedLanes()


    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def compareLanes(self, **kwargs):

        left_ckpts  = self.LeftLane.best_linepos if self.LeftLane.acceptPolynomial else self.LeftLane.current_linepos
        right_ckpts = self.RightLane.best_linepos if self.RightLane.acceptPolynomial else self.RightLane.current_linepos
        
        diff = right_ckpts - left_ckpts
        min_diff = np.round(diff.min(),0)
        max_diff = np.round(diff.max(),0)  
        diff_spread = round(max_diff - min_diff,0)

        diff_meters = np.round((np.array(right_ckpts)- np.array(left_ckpts))*self.LeftLane.MX,3)
        min_diff_meters = np.round(diff_meters.min(),3)
        max_diff_meters = np.round(diff_meters.max(),3)       
        diff_spread_meters = round(max_diff_meters - min_diff_meters,3)

        rejectedLane = self.LeftLane if self.RightLane.acceptPolynomial else self.RightLane 
        acceptedLane = self.RightLane if self.RightLane.acceptPolynomial else self.LeftLane
        
        print()
        print('compareLanes()') 
        print('                ', self.LeftLane.y_checkpoints)
        print(' left_ckpts    :', left_ckpts )
        print(' right_ckpts   :', right_ckpts)
        print(' diff (pixels) :', diff , 'Min: ', min_diff, ' Max: ', max_diff,  ' spread:', diff_spread)
        print(' diff (meters) :', diff_meters , 'Min: ', min_diff_meters, ' Max: ', max_diff_meters,  ' spread:', diff_spread_meters)

        if diff_spread < self.PARALLEL_LINES_MARGIN:
            print()
            print(' Spread between accepted lane ({}) and rejected lane ({}) is less than {} pixels - rejected lane will be accepted'.format(
                    acceptedLane.name, rejectedLane.name, self.PARALLEL_LINES_MARGIN))
            print()
            rejectedLane.acceptPolynomial = True
            rejectedLane.reset_best_fit(debug = self.debug)

        return 


    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def build_result_image(self, **kwargs):
        disp_start = kwargs.get('start' , self.displayRegionTop)
        disp_end   = kwargs.get('end'   , self.displayRegionBot)

        polyRegionColor1 = kwargs.get('polyRegionColor1', 'green')

        min_radius     = min(self.LeftLane.radius_history[-1], self.RightLane.radius_history[-1])
        min_radius_avg = min(self.LeftLane.radius_avg, self.RightLane.radius_avg)
    
        if  100 <= min_radius_avg < 125:
            disp_start +=   25
        elif  125 <= min_radius_avg < 200:
            disp_start +=   15
        elif 200 <= min_radius_avg < 250:
            disp_start +=   15
        elif 250 <= min_radius_avg < 300:
            disp_start +=   15            
        elif 300 <= min_radius_avg < 350:
            disp_start +=   10                
        elif 350 <= min_radius_avg < 400:
            disp_start +=   10
        elif 400 <= min_radius_avg < 450:
            disp_start +=   5
        elif 450 <= min_radius_avg < 500:
            disp_start +=   0
        else:   ## if min_radius_avg >    250:
            disp_start +=   0

        if self.debug:
            print('buildResultImage()')
            print('-'*15)
            print('  Hist LLane : ', [round(i,3) for i in self.LeftLane.radius_history[-10:]] )
            print('  Hist RLane : ', [round(i,3) for i in self.RightLane.radius_history[-10:]])
            # 0 print('Radius Diff History (m) : ', ['{:8.3f}'.format(i-j) for i,j in zip(RLane.radius, LLane.radius)])
            print('  Avg  LLane : [-5:] : {:8.0f}    [-10:] : {:8.0f} '.format(self.LeftLane.radius_avg, 
                    np.round(np.mean( self.LeftLane.radius_history[-10:]),3)))
            print('  Avg  RLane : [-5:] : {:8.0f}    [-10:] : {:8.0f} '.format(self.RightLane.radius_avg, 
                    np.round(np.mean(self.RightLane.radius_history[-10:]),3)))
            print(' Original disp_start : {:8d}      end: {:8d} '.format(self.displayRegionTop, self.displayRegionBot))
            print('       Min avg radius: {:8.0f}'.format( min_radius_avg))
            print(' Modified disp start : {:8d}      end: {:8d}'.format(disp_start, disp_end))
        
        self.curv_msg = curvatureMsg(self.LeftLane  , self.RightLane, debug = self.debug2)
        self.oc_msg   = offCenterMsg(self.LeftLane  , self.RightLane, self.camera_x, debug = self.debug2)
        thr_msg  = '{:5s} - {:22s}'.format(self.Conditions.upper(), self.thresholdMethod)
        stat_msg = 'RGB: {:3.0f}  Hue:{:3.0f}  SAT: {:3.0f}'.format(self.imgWarpedStats['RGB'][-1],
                    self.imgWarpedStats['Hue'][-1], self.imgWarpedStats['Sat'][-1])
         
        # if self.validLaneDetections:
        #     pass
        # else:
        #     beta = 0.3
            
        if True:
            self.resultImage, self.dyn_src_points_list = displayDetectedRegion(self.imgUndist, 
                                                                     self.LeftLane.fitted_best , 
                                                                     self.RightLane.fitted_best, 
                                                                     self.Minv, 
                                                                     disp_start = disp_start, 
                                                                     disp_end   = disp_end  ,
                                                                     alpha = 0.7,
                                                                     beta  = self.overlayBeta , 
                                                                     color = self.polyRegionColor1, 
                                                                     frameTitle = self.frameTitle, 
                                                                     debug = self.debug2)
        # else:
            # self.resultImage = np.copy(self.imgUndist)

        displayText(self.resultImage, 40, 40, self.frameTitle, fontHeight = 20)
        
        if self.validLaneDetections:
            displayText(self.resultImage, 40, 80, self.curv_msg  , fontHeight = 20)
            displayText(self.resultImage, 40,120, self.oc_msg    , fontHeight = 20)
        else:
            displayText(self.resultImage, 40, 80, 'Unable to detect lanes' , fontHeight = 20)

        displayText(self.resultImage, 850, 40, thr_msg  , fontHeight = 20)
        displayText(self.resultImage, 850, 80, stat_msg  , fontHeight = 20)

        # displayGuidelines(self.resultImage, draw = 'y');
        return


    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def adjust_RoI_window(self, **kwargs):
        '''
        Adjust the perspective transformation source points based on predefined criteria
        '''

        # min_radius    = min(self.LeftLane.radius[-1], self.RightLane.radius[-1])

        ### Build output image frame  
        
        mid_point_pixels    = self.LeftLane.line_base_pixels[-1] + (self.RightLane.line_base_pixels[-1] -self.LeftLane.line_base_pixels[-1]) / 2
        off_center_pixels   = round(self.camera_x - mid_point_pixels,0) 
        self.offctr_history.append(off_center_pixels)

        self.dyn_src_points = np.array(self.dyn_src_points_list, dtype = np.float32)
        diffs               = [abs(i[0] - j[0]) for i,j in zip(self.src_points_list[:2], self.dyn_src_points_list[:2])]
        max_diffs           = max(diffs)
        
        self.diffsSrcDynPoints.append(max_diffs)

        if self.debug:
            # np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =500, formatter = self.np_format)
            print()
            print('adjust_RoI_window() - FirstFrame:', self.firstFrame, ' AcceptPolynomial:', self.acceptPolynomials )
            print('-'*65)
            print('                 x_base :   Left: {:8.2f}    Right: {:8.2f}  '.format( self.LeftLane.x_base[-1],  self.RightLane.x_base[-1]))
            print('     Image Pixel Ratios :   Left: {:8.2f}    Right: {:8.2f}    Total: {:8.2f}'.format(
                    self.LeftLane.pixelRatio[-1], self.RightLane.pixelRatio[-1], self.imgPixelRatio[-1]))
                    
            # print('        Min last radius : {:7.0f}'.format( min_radius))        
            # print('            Left radius : {:7.2f}  History: {} '.format(self.LeftLane.radius[-1], self.LeftLane.radius[-10:])) 
            # print('           Right radius : {:7.2f}  History: {} '.format(self.RightLane.radius[-1], self.RightLane.radius[-10:])) 
            # print()
            print('      off center pixels : {:7.2f}  History: {} '.format(off_center_pixels, self.offctr_history[-10:]))
            print('     diff(dyn_src, src) : {:7.2f}  History: {} '.format(max_diffs, self.diffsSrcDynPoints[-10:]))
            print('    Pixel ratio - Left  : {:7.2f}  History: {} '.format( self.LeftLane.pixelRatio[-1],  self.LeftLane.pixelRatio[-10:]))
            print('    Pixel ratio - Right : {:7.2f}  History: {} '.format(self.RightLane.pixelRatio[-1], self.RightLane.pixelRatio[-10:]))
            print('    Pixel ratio - Image : {:7.2f}  History: {} '.format(self.imgPixelRatio[-1], self.imgPixelRatio[-10:]))
            print()        
            print('        src_points_list :  {} '.format(self.src_points_list))
            print('    dyn_src_points_list :  {} '.format(self.dyn_src_points_list))
            print('                  diffs :  {} '.format(diffs))
            print()

        if self.displayRealignment or self.debug:
            print('  Perspective transform source points -  OffCtr Pxls: {}   max source point diff: {}   OffCtr Threshold: {}   imgPxlRatio: {}  acceptCode: {}'.format(
                        off_center_pixels, max_diffs, self.OFF_CENTER_ROI_THRESHOLD, self.imgPixelRatio[-1], self.imgAcceptHistory[-1]))

        ###----------------------------------------------------------------------------------------------
        # if quality of last image threshold is > %80 and we need to run a bootstrap, set up to do so in
        # next video frame
        ###----------------------------------------------------------------------------------------------
        if  (self.acceptPolynomials) and \
             (( max_diffs >= self.OFF_CENTER_ROI_THRESHOLD )) :
            # or (self.firstFrame)):
            # ( ( max_diffs > self.CURRENT_OFFCTR_ROI_THR ) or (self.firstFrame)):
            

            if self.displayRealignment or self.debug:
                print()
                print('    Adjust perspective transform source points -  OffCtr Pxls: {}    max_diffs: {}    imgPxlRatio: {} '.format(
                        off_center_pixels, max_diffs, self.imgPixelRatio[-1]))
                print('   ','-'*100)
                print('    Cur src_points_list :  {} '.format(self.src_points_list))
                print()
                print('    New src_points_list :  {} '.format(self.dyn_src_points_list))
                print('       Prev Left x_base : ', self.LeftLane.x_base[-2], '   Right x_base  :', self.RightLane.x_base[-2])
                print('       New  Left x_base : ', self.LeftLane.x_base[-1], '   Right x_base  :', self.RightLane.x_base[-1])
                print()

                self.debugInfo_srcPointsRoI(title= 'source points prior to realignment')
                self.debugInfo_newSrcPointsRoI(title= 'new source points after realignment')


            self.prev_src_points_list = self.src_points_list
            self.src_points_list      = self.dyn_src_points_list
            self.src_points           = np.array(self.dyn_src_points_list, dtype = np.float32)

            self.slidingWindowBootstrap  = True
            self.RoIAdjustment           = True
            self.imgAdjustHistory.append((len(self.offctr_history), self.offctr_history[-1], self.diffsSrcDynPoints[-1]))
            # self.LeftLane.x_base.append (self.dyn_src_points_list[3][0])
            # self.RightLane.x_base.append(self.dyn_src_points_list[2][0])

            self.LeftLane.x_base.append (self.x_dst_left)
            self.RightLane.x_base.append(self.x_dst_right)
            # self.LeftLane.next_x_base = self.x_dst_left
            # self.RightLane.next_x_base = self.x_dst_right
        else:
            self.RoIAdjustment = False
        return 


    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def build_display_results(self, **kwargs):

        debug   = kwargs.get('debug', False)
        debug2  = kwargs.get('debug2', False)
        debug3  = kwargs.get('debug3', False)
        debug4  = kwargs.get('debug4', False)
        polyRegionColor1 = kwargs.get('color1', 'green')

        if debug:
            print(' Left lane MR fit           : ', self.LeftLane.proposed_fit , '    Right lane MR fit     : ', self.RightLane.proposed_fit)
            print(' Left lane MR best fit      : ', self.LeftLane.best_fit , '    Right lane MR best fit: ', self.RightLane.best_fit)
            print(' Left radius @ y =  10   : '+str(self.LeftLane.get_radius(10)) +" m   Right radius: "+str(self.RightLane.get_radius(10))+" m")
            print(' Left radius @ y = 700   : '+str(self.LeftLane.get_radius(700))+" m   Right radius: "+str(self.RightLane.get_radius(700))+" m")
            print(' Curvature message : ', curv_msg)
            print(' Off Center Message: ', oc_msg)            

        result_1, _  = displayDetectedRegion(self.imgUndist, self.LeftLane.proposed_curve, self.RightLane.proposed_curve, 
                                        self.Minv, disp_start= self.displayRegionTop , beta = 0.2, 
                                        color = self.polyRegionColor1, debug = False)

        displayText(result_1, 40, 40, self.frameTitle, fontHeight = 20)
        displayText(result_1, 40, 80, self.curv_msg, fontHeight = 20)
        displayText(result_1, 40,120, self.oc_msg, fontHeight = 20)
        # displayGuidelines(result_1, draw = 'y');

        ###----------------------------------------------------------------------------------------------
        ###  undistorted color image & perpective transformed image -- With RoI line display
        ###----------------------------------------------------------------------------------------------
        # imgRoI, imgRoIWarped = self.debugInfo_srcPointsRoI(display = False, title= 'Perspec. Tx. source points')            
        # imgLanePxls = self.visualizeLaneDetection(display = False)
            
        ###----------------------------------------------------------------------------------------------
        ## Certain operations are not performed based on the processing mode selected
        ##  Generate images for skipped operations for display purposes for display purposes
        ###----------------------------------------------------------------------------------------------
        if self.mode == 1:
            # print(' Display mode 1')
            ### results of applying thresholding AFTER warping undistorted image
            imgWarped, _, _  = perspectiveTransform(self.imgUndist, self.src_points, self.dst_points, debug = debug4)
            thresholdParms   = self.ImageThresholds[2][self.Conditions]
            output2          = apply_thresholds(self.imgWarped, thresholdParms, debug = debug2)
            self.imgWarpedThrshld = output2[self.thresholdMethod]
            self.imgThrshldWarped = self.working_image
        else: 
            # print(' Display mode 2')
            ### results of applying thresholding BEFORE warping undistorted image 
            thresholdParms   = self.ImageThresholds[1][self.Conditions] 
            output2          = apply_thresholds(self.imgUndist, thresholdParms, debug = debug2)  
            self.imgThrshld  = output2[self.thresholdMethod]
            self.imgThrshldWarped, _, _  = perspectiveTransform(self.imgThrshld, self.src_points, self.dst_points, debug = debug4) 
            self.imgWarpedThrshld = self.working_image
            
        self.resultExtraInfo = PlotDisplay(6,2)
        self.resultExtraInfo.addPlot(self.image       , title = 'original frame - '+self.frameTitle)
        self.resultExtraInfo.addPlot(self.imgUndist   , title = 'imgUndist - Undistorted Image')
        
        self.resultExtraInfo.addPlot(self.imgRoI      , title = 'imgRoI'   )
        self.resultExtraInfo.addPlot(self.imgRoIWarped, title = 'imgRoIWarped' )
        
        self.resultExtraInfo.addPlot(self.imgThrshld  , title = 'imgThrshld - Thresholded image')
        self.resultExtraInfo.addPlot(self.imgWarped   , title = 'imgWarped - Warped Image')
        
        self.resultExtraInfo.addPlot(self.imgThrshldWarped, title = 'imgThrshldWarped - Img Thresholded ---> Warped (Mode 1)')
        self.resultExtraInfo.addPlot(self.imgWarpedThrshld, title = 'imgWarpedThrshld - Img Warped ---> Thresholded (Mode 2)')
        
        self.resultExtraInfo.addPlot(self.imgLanePxls , title = 'ImgLanePxls (Black: Prev fit, Yellow: New fit, Red: Best Fit)' )
        self.resultExtraInfo.addPlot(self.histogram   , title = 'Histogram of activated pixels', type = 'plot' )
        
        self.resultExtraInfo.addPlot(result_1         , title = 'result_1 : Using LAST fit')
        self.resultExtraInfo.addPlot(self.resultImage , title = 'finalImage : Using BEST fit'+self.frameTitle)
        self.resultExtraInfo.closePlot()

        return 
    

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
        self.slidingWindowBootstrap = True
        self.firstFrame             = True

        self.LeftLane     = Line(history = self.HISTORY, height = self.height, y_src_top = self.y_src_top, y_src_bot = self.y_src_bot)
        self.RightLane    = Line(history = self.HISTORY, height = self.height, y_src_top = self.y_src_top, y_src_bot = self.y_src_bot)
        return True


    def compute_top_x_disp(self):        
        
        top_left_x = ((self.y_src_bot - self.y_src_top) / self.tan_theta) +  (self.x_src_center - self.x_bot_disp)
        
        top_x_disp = int(round(self.x_src_center - top_left_x,0))
        
        print('self.x_src_top_left: ',top_left_x, '   top_x_disp: ', top_x_disp)
        return top_x_disp


    def build_source_RoI_region(self):
        self.x_src_center  = 640 + self.RoI_x_adj        
        self.tan_theta = (self.lane_theta * np.pi)/180    
        
        self.x_top_disp               =  self.compute_top_x_disp()

        self.x_src_bot_left           =  self.x_src_center - self.x_bot_disp     # = 295) 
        self.x_src_bot_right          =  self.x_src_center + self.x_bot_disp     # = 1105) 
        self.x_src_top_left           =  self.x_src_center - self.x_top_disp     # = 600)  ## 580 -> 573
        self.x_src_top_right          =  self.x_src_center + self.x_top_disp     # = 740)

        src_points_list               =   [ (self.x_src_top_left , self.y_src_top),
                                            (self.x_src_top_right, self.y_src_top), 
                                            (self.x_src_bot_right, self.y_src_bot),
                                            (self.x_src_bot_left , self.y_src_bot)]

        src_points_array              = np.array(src_points_list, dtype = np.float32)
        return src_points_list, src_points_array


    def build_dest_RoI_region(self):
        dst_points_list               =  [ (self.x_dst_left     , self.y_dst_top), 
                                           (self.x_dst_right    , self.y_dst_top), 
                                           (self.x_dst_right    , self.y_dst_bot), 
                                           (self.x_dst_left     , self.y_dst_bot)]
    
        dst_points_array              = np.array(dst_points_list, dtype = np.float32)

        return dst_points_list, dst_points_array


    def set_thresholding_parms(self):
        '''
        select thresholding parameters based on current image condtiions
        currently we only compare the RGB mean value against a threshold
        other criteria can be considered
        '''

        if (self.imgWarpedStats['Sat'][-1] > self.XHIGH_SAT_THRESHOLD) or \
           (self.imgWarpedStats['RGB'][-1] > self.HIGH_RGB_THRESHOLD):
                self.Conditions = 'xhigh'
                historyFlag = 30

        elif (self.imgWarpedStats['RGB'][-1] < self.VLOW_RGB_THRESHOLD) :
            self.Conditions = 'vlow'
            historyFlag = -20
        
        elif (self.imgWarpedStats['RGB'][-1] < self.LOW_RGB_THRESHOLD) :
          
            if (self.imgWarpedStats['Sat'][-1] < self.LOW_SAT_THRESHOLD):
                self.Conditions = 'lowsat'
                historyFlag = -30
            elif (self.imgWarpedStats['Sat'][-1] > self.HIGH_SAT_THRESHOLD):
                self.Conditions = 'hisat'
                historyFlag = +20
            else:
                self.Conditions = 'low'
                historyFlag = -10                
        
        elif (self.imgWarpedStats['RGB'][-1] < self.MED_RGB_THRESHOLD) :
          
            if (self.imgWarpedStats['Sat'][-1] > self.HIGH_SAT_THRESHOLD):
                self.Conditions = 'hisat'
                historyFlag = 20
            # if (self.imgWarpedStats['Sat'][-1] < self.LOW_SAT_THRESHOLD):
                # self.Conditions = 'lowsat'
                # historyFlag = -30
            else:
                self.Conditions = 'med'
                historyFlag = 0

        # elif  (self.imgWarpedStats['RGB'][-1] < self.HIGH_RGB_THRESHOLD) :
        else: 
            if (self.imgWarpedStats['Sat'][-1] > self.HIGH_SAT_THRESHOLD):
                self.Conditions = 'hisat'
                historyFlag = 20
            else:
                self.Conditions = 'high'
                historyFlag = 10

        self.imgThrshldHistory.append(historyFlag)
        self.thresholdMethod  =  self.thresholdMethods[self.mode][self.Conditions]
        self.thresholdStrs    =  self.itStr[self.mode][self.Conditions]           
        self.thresholdParms   =  self.ImageThresholds[self.mode][self.Conditions]         
    
        return 


    def initialize_thresholding_parameters(self):
        ##---------------------------------------------
        ## Image Thresholding params 
        ##---------------------------------------------
        self.ImageThresholds  = defaultdict(dict) ## { 1: {} , 2: {} }
        self.itStr            = defaultdict(dict) ## { 1: {} , 2: {} }
        self.thresholdMethods = defaultdict(dict) ## { 1: {} , 2: {} }

        self.thresholdMethods[1]['xhigh']  =  self.XHIGH_THRESHOLDING   
        self.thresholdMethods[1]['high']   =  self.HIGH_THRESHOLDING   
        self.thresholdMethods[1]['med']    =  self.NORMAL_THRESHOLDING 
        self.thresholdMethods[1]['low']    =  self.LOW_THRESHOLDING    
        self.thresholdMethods[1]['vlow']   =  self.VLOW_THRESHOLDING    
        self.thresholdMethods[1]['hisat']  =  self.HISAT_THRESHOLDING  
        self.thresholdMethods[1]['lowsat'] =  self.LOWSAT_THRESHOLDING 

        ## Normal Light Conditions ------------
        self.ImageThresholds[1]['xhigh'] = {
            'ksize'      : 7         ,
            'x_thr'      : (30,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (35,255)  ,
            'dir_thr'    : (40,65)   ,
            'sat_thr'    : (110,255) ,
            'lvl_thr'    : (205, 255),
            'rgb_thr'    : (205,255) ,
            'hue_thr'    : None 
        }
        self.ImageThresholds[1]['high'] = {
            'ksize'      : 7         ,
            'x_thr'      : (30,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (35,255)  ,
            'dir_thr'    : (40,65)   ,
            'sat_thr'    : (110,255) ,
            'lvl_thr'    : (205,255),
            'rgb_thr'    : (205,255) ,
            'hue_thr'    : None 
        }

        self.ImageThresholds[1]['med'] = {
            'ksize'      : 7         ,
            'x_thr'      : (30,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (35,255)  ,
            'dir_thr'    : (40,65)   ,
            'sat_thr'    : (110,255) ,
            'lvl_thr'    : (205,255),
            'rgb_thr'    : (205,255) ,
            'hue_thr'    : None 
        }
        ## Dark Light Conditions ------------
        self.ImageThresholds[1]['low']= {
            'ksize'      : 7         ,
            'x_thr'      : ( 30,255) ,
            'y_thr'      : ( 30,255) ,   ## changed from ( 30,255)  2-26-20
            'mag_thr'    : ( 35,255) ,
            'dir_thr'    : ( 40, 65) ,
            'sat_thr'    : (160,255) ,   ## changed from (110,255)  2-26-20
            'lvl_thr'    : (205,255) ,
            'rgb_thr'    : (205,255) ,
            'hue_thr'    : None  
        }
        ## Dark Light Conditions ------------
        self.ImageThresholds[1]['vlow']= {
            'ksize'      : 7         ,
            'x_thr'      : ( 30,255) ,
            'y_thr'      : ( 30,255) ,   ## changed from ( 30,255)  2-26-20
            'mag_thr'    : ( 35,255) ,
            'dir_thr'    : ( 40, 65) ,
            'sat_thr'    : (160,255) ,   ## changed from (110,255)  2-26-20
            'lvl_thr'    : (205,255) ,
            'rgb_thr'    : (205,255) ,
            'hue_thr'    : None  
        }
        self.ImageThresholds[1]['hisat'] = {
            'ksize'      : 7         ,
            'x_thr'      : (30,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (35,255)  ,
            'dir_thr'    : (40,65)   ,
            'sat_thr'    : (110,255) ,
            'lvl_thr'    : (205, 255),
            'rgb_thr'    : (205,255) ,
            'hue_thr'    : None 
        }
        self.ImageThresholds[1]['lowsat']= {
            'ksize'      : 7         ,
            'x_thr'      : (45,255)  ,
            'y_thr'      : None      ,
            'mag_thr'    : None      ,   ### (25,250)  ,
            'dir_thr'    : None      ,
            'sat_thr'    : None      ,
            'lvl_thr'    : None      ,
            'rgb_thr'    : None      ,
            'hue_thr'    : ( 15, 50)
        }

 
        ##------------------------------------
        ## Warped Image Threshold params
        ##------------------------------------

        self.thresholdMethods[2]['xhigh']  =  self.XHIGH_THRESHOLDING   
        self.thresholdMethods[2]['high']   =  self.HIGH_THRESHOLDING   
        self.thresholdMethods[2]['med']    =  self.NORMAL_THRESHOLDING 
        self.thresholdMethods[2]['low']    =  self.LOW_THRESHOLDING    
        self.thresholdMethods[2]['vlow']   =  self.VLOW_THRESHOLDING    
        self.thresholdMethods[2]['hisat']  =  self.HISAT_THRESHOLDING  
        self.thresholdMethods[2]['lowsat'] =  self.LOWSAT_THRESHOLDING         

        self.ImageThresholds[2]['xhigh'] = {
            'ksize'      : 7         , 
            'x_thr'      : (30,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (10,50)   ,
            'dir_thr'    : (0,30)    ,
            'sat_thr'    : (60, 255) ,   ### (80, 255) ,
            'lvl_thr'    : (180,255) ,
            'rgb_thr'    : (180,255) ,
            'hue_thr'    : None 
        }
        ## Normal Light Conditions ------------
        self.ImageThresholds[2]['high'] = {
            'ksize'      : 7         , 
            'x_thr'      : (30,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (10,50)   ,
            'dir_thr'    : (0,30)    ,
            'sat_thr'    : (60, 255) ,   ### (80, 255) ,
            'lvl_thr'    : (180,255) ,
            'rgb_thr'    : (180,255) ,
            'hue_thr'    : None 
        }

        self.ImageThresholds[2]['med'] = {
            'ksize'      : 7         , 
            'x_thr'      : (30,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (10,50)   ,
            'dir_thr'    : (0,30)    ,
            'sat_thr'    : (60, 255) ,   ### (80, 255) ,
            'lvl_thr'    : (180,255) ,
            'rgb_thr'    : (180,255) ,
            'hue_thr'    : None 
        }
        ## dark conditions--------------
        self.ImageThresholds[2]['low'] = {
            'ksize'      : 7         ,
            'x_thr'      : (70,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (5, 100)  ,   ### (25,250)  ,
            'dir_thr'    : (0,30)    ,
            'sat_thr'    : (130,255) ,
            'lvl_thr'    : (200,255) ,
            'rgb_thr'    : (200,255) ,
            'hue_thr'    : ( 15, 50)            
        }
        ## dark conditions--------------
        self.ImageThresholds[2]['vlow'] = {
            'ksize'      : 7         ,
            'x_thr'      : (70,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (5, 100)  ,   ### (25,250)  ,
            'dir_thr'    : (0,30)    ,
            'sat_thr'    : (130,255) ,
            'lvl_thr'    : (200,255) ,
            'rgb_thr'    : (200,255) ,
            'hue_thr'    : ( 15, 50)            
        }

        self.ImageThresholds[2]['hisat'] = {
            'ksize'      : 7         , 
            'x_thr'      : (30,255)  ,
            'y_thr'      : (70,255)  ,
            'mag_thr'    : (10,50)   ,
            'dir_thr'    : (0,30)    ,
            'sat_thr'    : (60, 255) ,   ### (80, 255) ,
            'lvl_thr'    : (180,255) ,
            'rgb_thr'    : (180,255) ,
            'hue_thr'    : None 
        }

        self.ImageThresholds[2]['lowsat']= {
            'ksize'      : 7         ,
            'x_thr'      : (45,255)  ,
            'y_thr'      : None      ,
            'mag_thr'    : None      ,   ### (25,250)  ,
            'dir_thr'    : None      ,
            'sat_thr'    : None      ,
            'lvl_thr'    : None      ,
            'rgb_thr'    : None      ,
            'hue_thr'    : ( 15, 50)
        }

        self.thresholds_to_str()


    def thresholds_to_str(self, debug = False):
        for mode in [1,2]:
            for cond in self.ImageThresholds[mode].keys():
                if debug:
                    print(mode ,    ' Threshold key: ',cond)
                self.itStr[mode][cond] = {}
                for thr in self.ImageThresholds[mode][cond].keys():
                    self.itStr[mode][cond][thr] = str(self.ImageThresholds[mode][cond][thr])
                    if debug:
                        print('      thr : ', thr, '   ', self.ImageThresholds[mode][cond][thr])


    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def debugInfo_DetectedLanes(self, display = 3, size = (24,9)):
                    
        self.prevBestFit  = colorLanePixels(self.out_img, self.LeftLane, self.RightLane)

        if self.HISTORY > 1:
            self.prevBestFit = displayPolynomial(self.prevBestFit, self.LeftLane.fitted_best_history, self.RightLane.fitted_best_history, 
                                                 iteration = -2,  color = 'aqua')
            self.prevBestFit = displayPolynomial(self.prevBestFit, self.LeftLane.proposed_curve, self.RightLane.proposed_curve, 
                                                 iteration = -1,  color = 'yellow')

        # currentFit  = displayPolynomial(prevBestFit, self.LeftLane.proposed_curve, self.RightLane.proposed_curve, iteration = -1, color = 'yellow')

        self.imgLanePxls = displayPolynomial(self.prevBestFit, self.LeftLane.fitted_best, self.RightLane.fitted_best, color = 'fuchsia', thickness = 2)
        if display:
            # print(' y_src_top_left : {}   y_src_top_right: {}   y_src_bot_left: {}   y_src_bot_right: {}'.format(self.dst_points_list))
                # self.y_src_top, self.y_src_top, self.y_src_bot, self.y_src_bot))  
            if display in [1,3]:
                print('      x_src_top_left : {}   x_src_top_right: {}   x_src_bot_left: {}   x_src_bot_right: {}'.format(
                        self.src_points_list[0], self.src_points_list[1],self.src_points_list[3],self.src_points_list[2]))
                display_two(self.working_image, self.out_img, size = size, title1 = 'working_image - '+self.frameTitle, 
                            title2 = 'out_img ')
            if display in [2,3]:
                display_two(self.prevBestFit, self.imgLanePxls, size = size, title1 = 'Prev best fit (Cyan: Prev fit, Yellow: New proposal)' , 
                            title2 = 'ImgLanePxls (Cyan: Prev fit, Yellow: New proposal, Fuschia: New Best Fit)' )
            print()
        
        return 


    def debugInfo_ImageSummaryInfo(self):
        print('Frame: {:4d} - {:.0f} ms - Image RGB: {:3.0f}  ({:3.0f},{:3.0f},{:3.0f})     '\
                '    WARPED RGB: {:3.0f}  HLS: {:3.0f}   H: {:3.0f}   L: {:3.0f}   S: {:3.0f}'\
                '    {:5s} - {:10s}'.format(self.inVideo.currFrameNum, self.inVideo.currPos,
                self.imgUndistStats['RGB'][-1],  
                self.imgUndistStats['Red'][-1], self.imgUndistStats['Grn'][-1], self.imgUndistStats['Blu'][-1],
                self.imgWarpedStats['RGB'][-1], self.imgWarpedStats['HLS'][-1], 
                self.imgWarpedStats['Hue'][-1], self.imgWarpedStats['Lvl'][-1], self.imgWarpedStats['Sat'][-1],     
                self.Conditions.upper(), self.thresholdMethod))
        if self.debug:
            print( ' Thresholds:  HIGH RGB: {}    MED RGB: {}   LOW RGB: {}  VLOW RGB: {}   X-HIGH SAT: {}  HIGH SAT: {}   LOW SAT: {} '.
                format(self.HIGH_RGB_THRESHOLD, self.MED_RGB_THRESHOLD , self.LOW_RGB_THRESHOLD, 
                       self.VLOW_RGB_THRESHOLD, self.XHIGH_SAT_THRESHOLD, self.HIGH_SAT_THRESHOLD, self.LOW_SAT_THRESHOLD))                
        return


    def debugInfo_ImageInfo(self, frame = -1):
        print('Frame: {:4.0f} - Mode: {:2d}  imgUndist -  Avgs  RGB: {:6.2f}  HLS:{:6.2f}  Sat: {:6.2f}  Hue: {:6.2f}  Lvl: {:6.2f}'\
                ' -- {:5s} - {:10s}'.format( self.inVideo.currFrameNum, self.mode,  
                self.imgUndistStats['RGB'][frame], self.imgUndistStats['HLS'][frame], 
                self.imgUndistStats['Sat'][frame], self.imgUndistStats['Hue'][frame], 
                self.imgUndistStats['Lvl'][frame], self.Conditions ,  self.thresholdMethod))
        
        print(' {:22s} imgWarped -  Avgs  RGB: {:6.2f}  HLS:{:6.2f}  Sat: {:6.2f}  Hue: {:6.2f}  Lvl: {:6.2f}'\
                ' -- {:5s} - {:10s}'.format( '',  
                self.imgWarpedStats['RGB'][frame], self.imgWarpedStats['HLS'][frame], 
                self.imgWarpedStats['Sat'][frame], self.imgWarpedStats['Hue'][frame], 
                self.imgWarpedStats['Lvl'][frame], self.Conditions ,  self.thresholdMethod))
        
        display_multi(self.inVideo.image, self.imgUndist, self.imgWarped, title3 = 'Warped', grid2 = 'minor')
        return


    def debugInfo_LaneDetInfo(self):
        imgPixelRatio, NztoSrchNzRatio, NztoImageNzRatio, ttlImageNZPixels, ttlLaneNZPixels = self.detStats
        print('  NZ pixels  - in image  : {:8d}   search reg: {:8d}  '\
              '    Nz to imgPixel Ratio: %{:5.2f}    Nz to SrchRegion Ratio : %{:5.2f}    Nz to ImageNz Ratio: %{:5.2f}' .
              format(ttlImageNZPixels, ttlLaneNZPixels, imgPixelRatio   , NztoSrchNzRatio, NztoImageNzRatio))
        print('  Detected Pixel Count L : {:8d}   R         : {:8d}      Detected Pixel Ratio  L: %{:5.2f}    R: %{:5.2f} '.
              format(self.LeftLane.pixelCount[-1], self.RightLane.pixelCount[-1],
                     self.LeftLane.pixelRatio[-1], self.RightLane.pixelRatio[-1]))
        return 


    def debugInfo_ThresholdedImage(self):

        display_two(self.imgThrshld, self.working_image, title1 = self.thresholdMethod +' '+str(np.sum(self.imgThrshld)), 
                                                         title2 = 'After thresholding - '+str(np.sum(self.working_image)))
        return 


    def debugInfo_srcPointsRoI(self, size = (24,9), title = None ):
        print()
        print('   x_top_disp     : {:<13d}  x_src_center    : {:<13d}  x_bot_disp     : {:<4d} '.format(
                  self.x_top_disp, self.x_src_center, self.x_bot_disp))
        print('   x_src_top_left : {:12s}   x_src_top_right : {:12s}   x_src_bot_left : {:12s}   x_src_bot_right : {:12s}'.format(
                  str(self.src_points_list[0]), str(self.src_points_list[1]), str(self.src_points_list[3]), str(self.src_points_list[2])))
        print('   y_src_top_left : {:12s}   y_src_top_right : {:12s}   y_src_bot_left : {:12s}   y_src_bot_right : {:12s}'.format(
                  str(self.dst_points_list[0]), str(self.dst_points_list[1]), str(self.dst_points_list[3]), str(self.dst_points_list[2])))  
        
        display_two(self.imgRoI  , self.imgRoIWarped, title1 = title , grid1 = 'major',
                                            title2 = title + ' - after perspective transformation', grid2 = 'major', size = size)
        print()
        return  


    def debugInfo_newSrcPointsRoI(self, display = True, size = (24,9), title = None):
        imgRoI             = displayRoILines(self.imgUndist, self.dyn_src_points_list , color = 'blue', thickness = 2)
        imgRoIWarped, _, _ = perspectiveTransform(imgRoI   , self.dyn_src_points      , self.dst_points)
        imgRoIWarped       = displayRoILines(imgRoIWarped  , self.dst_points_list     , thickness = 2, color = 'yellow')
        
        display_two(imgRoI  , imgRoIWarped   , title1 = title , grid1 = 'major',
                                               title2 = title+' - after perspective transformation ' , grid2 = 'major', size = size)
        return imgRoI, imgRoIWarped 


    def debugInfo_DetectionTransform(self):
        self.debugInfo_DetectedLanes(display=0)

        # imgWarped, _, _ = perspectiveTransform(imgUnwarped   , self.dyn_src_points      , self.dst_points)
        imgWarped = cv2.warpPerspective(self.imgLanePxls, self.Minv, self.imgLanePxls.shape[1::-1], flags=cv2.INTER_LINEAR)
        display_two(self.imgLanePxls, imgWarped, title1 = 'Detection prewarped' , grid1 = 'minor',
                                                 title2 = ' Detection - Warped' , grid2 = 'major', size = (24,9))
        return  


    def debugInfo_RoITransforms(self):
        self.debugInfo_srcPointsRoI(title= 'source points prior to realignment')
        self.debugInfo_newSrcPointsRoI()
        return 


    def display_thresholds(self, mode = None):
        if mode is None:
            mode = [self.mode]
        if isinstance(mode, int):
            mode = [mode]
        print()

        print( ' Thresholds:  HIGH RGB: {}    MED RGB: {}   LOW RGB: {}  VLOW RGB: {}   XHIGH SAT: {}   HIGH SAT: {}   LOW SAT: {} '.format(
                self.HIGH_RGB_THRESHOLD , self.MED_RGB_THRESHOLD , self.LOW_RGB_THRESHOLD, self.VLOW_RGB_THRESHOLD, 
                self.XHIGH_SAT_THRESHOLD, self.HIGH_SAT_THRESHOLD, self.LOW_SAT_THRESHOLD))

        print()
        for mod in mode:
            print('','-'*150)
            print(' | {:8s} | {:^20s} | {:^20s} | {:^20s} | {:^20s} | {:^20s} | {:^20s} |'.format('',
            'PL[X-High]','PL[High]','PL[Med]','PL[Low]','PL[VLow]','PL[HiSat]','PL[LoSat]'))

            print(' | {:8s} | RGB> {:<3d} or SAT > {:<3d}|  {:>5d} > RGB > {:<5d} |  {:>5d} > RGB > {:<5d} |  {:>5d} > RGB > {:<5d} |'\
            ' {:>8s} > {:<9d} | {:>8s} > {:<9d} |'.format('', self.HIGH_RGB_THRESHOLD, self.XHIGH_SAT_THRESHOLD, 
            self.HIGH_RGB_THRESHOLD, self.MED_RGB_THRESHOLD,   self.MED_RGB_THRESHOLD , self.LOW_RGB_THRESHOLD ,
            self.LOW_RGB_THRESHOLD , self.VLOW_RGB_THRESHOLD, 'SAT', self.HIGH_SAT_THRESHOLD, 'SAT', self.LOW_SAT_THRESHOLD))
            print('','-'*150)
            print(' | Mode{:2d}   : {:^20s} | {:^20s} | {:^20s} | {:^20s} | {:^20s} | {:^20s} |'.format(mod,
                        self.thresholdMethods[mod]['xhigh'], self.thresholdMethods[mod]['high'], 
                        self.thresholdMethods[mod]['med']  , self.thresholdMethods[mod]['low'], 
                        self.thresholdMethods[mod]['vlow'] , self.thresholdMethods[mod]['hisat']))
            print('','-'*150)

            for ke in self.ImageThresholds[mod]['xhigh'].keys():
                print(' | {:8s} : {:^20s} | {:^20s} | {:^20s} | {:^20s} | {:^20s} | {:^20s} |'.format(ke, 
                    str(self.ImageThresholds[mod]['xhigh'][ke]) ,                                                                                                                              
                    str(self.ImageThresholds[mod]['high'][ke])  , 
                    str(self.ImageThresholds[mod]['med'][ke])   , 
                    str(self.ImageThresholds[mod]['low'][ke])   , 
                    str(self.ImageThresholds[mod]['vlow'][ke])  , 
                    str(self.ImageThresholds[mod]['hisat'][ke]), 
                    str(self.ImageThresholds[mod]['lowsat'][ke]) 
                ))    
            print('','-'*150)
            print()
        return 


    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def debugInfo_displayFittingInfo(self):
        np_format = {}
        np_format['float'] = lambda x: "%8.2f" % x
        np_format['int']   = lambda x: "%8d" % x
        np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =100, formatter = np_format)
        
        print()
        print('='*70)
        print('Display fitting info for ', self.frameTitle)
        print('='*70)

        print()
        print('Proposed Polynomial      left :  {}    right : {} '.format(self.LeftLane.proposed_fit, self.RightLane.proposed_fit))
        print('Best Fit Polynomial      left :  {}    right : {} '.format(self.LeftLane.best_fit, self.RightLane.best_fit))
        print('Diff(proposed,best_fit)  left :  {}    right : {} '.format( self.LeftLane.best_fit-self.LeftLane.proposed_fit, 
                                                                                self.RightLane.best_fit-self.RightLane.proposed_fit))
        print('RSE(Proposed,best fit):  left :  {:<30.3f}    right : {:<30.3f} '.format(self.LeftLane.RSE ,self.RightLane.RSE ))
        print()


        # print()
        # print('Proposed Polynomial:')
        # print('-'*40)
        # print('left      : {}    right     : {} '.format(self.LeftLane.proposed_fit, self.RightLane.proposed_fit))

        # if len(self.LeftLane.proposed_fit_history) > 1:
        print()
        print('Best Fit Polynomials:')
        print('-'*40)

        for idx in range(-1, -min(len(self.LeftLane.best_fit_history), self.HISTORY+1) , -1):
            ls, rs = self.LeftLane.best_fit_history[idx], self.RightLane.best_fit_history[idx]
            print('left[{:2d}]    : {}    right[{:2d}]     : {} '.format(idx,ls, idx,rs))

        #     print()
        #     print('Diff b/w proposed and best_fit polynomial ')
        #     print('-'*40)
        #     print('left      : {}    right     : {} '.format( self.LeftLane.best_fit-self.LeftLane.proposed_fit, 
        #                                                     self.RightLane.best_fit-self.RightLane.proposed_fit)    )
        #     print()
        #     print('Proposed RSE with best fit - self.LeftLane: {}   RLane :  {} '.format(self.LeftLane.RSE ,self.RightLane.RSE ))
        #     print()
        #
        #     print('Best RSE Hist LLane : ',  self.LeftLane.RSE_history[-15:])
        #     print('Best RSE Hist RLane : ', self.RightLane.RSE_history[-15:])
        #     print('Best fit RSE Hist LeftLane  : ', ['{:8.3f}'.format(i) for i in self.LeftLane.RSE_history])
        #     print('Best fit RSE Hist RightLane : ', ['{:8.3f}'.format(i) for i in self.RightLane.RSE_history])
            
        print()
        print('-'*40)
        print('Previously proposed Polynomials:')
        print('-'*40)
        
        for idx in range(-1, -min(len(self.LeftLane.proposed_fit_history), self.HISTORY+1) , -1):
            ls, rs = self.LeftLane.proposed_fit_history[idx], self.RightLane.proposed_fit_history[idx]
            print('left[{:2d}]    : {}    right[{:2d}]     : {} '.format(idx,ls, idx,rs))
        
        print()
        print('RSE History - Left  : ',  self.LeftLane.RSE_history[-15:])
        print('RSE History - Right : ', self.RightLane.RSE_history[-15:])
        
        # print('fit RSE Hist LeftLane  : ',  self.LeftLane.RSE_history[-15:])
        # print('fit RSE Hist RightLane : ', self.RightLane.RSE_history[-15:])
        # print('fit RSE Hist RightLane : ', ['{:8.3f}'.format(i) for i in self.RightLane.RSE_history])
        # print('fit RSE Hist LeftLane  : ', ['{:8.3f}'.format(i) for i in  self.LeftLane.RSE_history])


        ###--------------------------------------------------------------------------------------
        ###  Radius of Curvature
        ###--------------------------------------------------------------------------------------
        print()
        print('-'*40)    
        print('Lane Radius from proposed fit:')
        print('-'*40)
        ls = self.LeftLane.current_radius
        rs = self.RightLane.current_radius
        diff = np.round(np.array(rs)- np.array(ls),3)
        avg  = np.round((np.array(rs) + np.array(ls))/2,3)
        print('       Y   : ', self.LeftLane.y_checkpoints)
        print('left       : ',ls  , '  Avg:', np.round(np.mean(ls),3))
        print('right      : ',rs  , '  Avg:', np.round(np.mean(rs),3))
        print()
        print('avg        : ',avg , '  Avg:', np.round(np.mean(avg),3))
        print('diff       : ',diff, '  Max:', diff.max())

        if (self.HISTORY > 1)  and  len(self.LeftLane.best_fit_history) > 0:
            print()
            print('Lane Radius from BEST line fit:')
            print('-'*40)
            ls =  self.LeftLane.best_radius
            rs = self.RightLane.best_radius
            diff = np.round(np.array(rs)- np.array(ls),3)
            avg  = np.round((np.array(rs) + np.array(ls))/2,3)
            print('       Y  : ', self.LeftLane.y_checkpoints)
            print('left      : ', ls  , '  Avg:', np.round(np.mean(ls),3))
            print('right     : ', rs  , '  Avg:', np.round(np.mean(rs),3))
            print()
            print('avg       : ', avg , '  Avg:', np.round(np.mean(avg),3))
            print('diff      : ', diff, '  Max:', diff.max())
        
        print()
        print('Hist LLane : ', [round(i,3) for i in self.LeftLane.radius_history[-10:]] )
        print('Hist RLane : ', [round(i,3) for i in self.RightLane.radius_history[-10:]])
        # print('Radius Diff History (m) : ', ['{:8.3f}'.format(i-j) for i,j in zip(RLane.radius, LLane.radius)])
        print('Avg  LLane : past 5 frames: {:8.3f}    past 10 frames: {:8.3f} '.format(self.LeftLane.radius_avg, 
                    np.round(np.mean( self.LeftLane.radius_history[-10:]),3)))
        print('Avg  RLane : past 5 frames: {:8.3f}    past 10 frames: {:8.3f} '.format(self.RightLane.radius_avg, 
                    np.round(np.mean(self.RightLane.radius_history[-10:]),3)))
        

        ###--------------------------------------------------------------------------------------
        ###  Lane Slope
        ###--------------------------------------------------------------------------------------
        print()
        print('-'*40)
        print('Lane Slopes from latest proposed fit:')
        print('-'*40)
        ls = self.LeftLane.current_slope
        rs = self.RightLane.current_slope
        diff = np.round(np.array(rs)- np.array(ls),3)
        avg  = np.round((np.array(rs) + np.array(ls))/2,3)
        print('       Y  : ', self.LeftLane.y_checkpoints)
        print('left      : ',ls  , ' Avg:', np.round(np.mean(ls),3))
        print('right     : ',rs  , ' Avg:', np.round(np.mean(rs),3))
        print()
        print('avg       : ',avg)
        print('diff      : ',diff, ' Min/Max[700:480]: ', diff[0:5].min(), diff[0:5].max())

        if len(self.LeftLane.best_fit_history) > 0:
            print()
            print('Lane Slopes from BEST fit:') 
            print('-'*40)
            ls = self.LeftLane.best_slope
            rs = self.RightLane.best_slope
            diff = np.round(np.array(rs)- np.array(ls),3)
            avg  = np.round((np.array(rs) + np.array(ls))/2,3)
            print('       Y  : ', self.LeftLane.y_checkpoints)
            print('left      : ',ls  , ' Avg:', np.round(np.mean(ls),3))
            print('right     : ',rs  , ' Avg:', np.round(np.mean(rs),3))
            print()
            print('avg       : ',avg)
            print('diff      : ',diff, ' Min/Max[700:480]: ', diff[0:5].min(), diff[0:5].max())

        print()
        print('Slope Hist LLane : ', [round(i,3) for i in self.LeftLane.slope[-10:]])
        print('Slope Hist RLane : ', [round(i,3) for i in self.RightLane.slope[-10:]])
        print('Slope Diff Hist  : ', [round(i-j,3) for i,j in zip(self.RightLane.slope[-10:], self.LeftLane.slope[-10:])])
        
        ###--------------------------------------------------------------------------------------
        ###  Lanes X position - Current frame
        ###--------------------------------------------------------------------------------------
        print()
        print('-'*40)
        print('Line X Position - PROPOSED FIT:')
        print('-'*40)
        ls = self.LeftLane.current_linepos
        rs = self.RightLane.current_linepos
        ls_min =  ls.min()
        ls_max =  ls.max()
        rs_min =  rs.min()
        rs_max =  rs.max()
        diff_pxl = np.round( np.array(rs)- np.array(ls),3)
        diff_pxl_min =  diff_pxl.min()
        diff_pxl_max =  diff_pxl.max()
        diff_mtr = np.round((np.array(rs)- np.array(ls))*self.LeftLane.MX,3)
        diff_mtr_min =  diff_mtr.min()
        diff_mtr_max =  diff_mtr.max()
        print('        Y : {}'.format(self.LeftLane.y_checkpoints))
        print(' Left   X : {}  Min: {:8.3f}  Max: {:8.3f}  spread{:8.3f}'.format(ls, ls_min, ls_max, ls_max - ls_min))
        print(' Right  X : {}  Min: {:8.3f}  Max: {:8.3f}  spread{:8.3f}'.format(rs, rs_min, rs_max, rs_max - rs_min))
        print('\n diff pxl : {}  Min: {:8.3f}  Max: {:8.3f}  spread{:8.3f}'.format(diff_pxl, diff_pxl_min, diff_pxl_max, diff_pxl_max - diff_pxl_min))
        print(' diff mtr : {}  Min: {:8.3f}  Max: {:8.3f}  spread{:8.3f}'.format(diff_mtr, diff_mtr_min, diff_mtr_max, diff_mtr_max - diff_mtr_min))


        if len(self.LeftLane.best_fit_history) > 0:
            print()
            print('-'*40)
            print('Line X Position - BEST FIT:')
            print('-'*40)
            ls = self.LeftLane.best_linepos
            rs = self.RightLane.best_linepos
            ls_min =  ls.min()
            ls_max =  ls.max()
            rs_min =  rs.min()
            rs_max =  rs.max()
            diff_pxl = np.round( np.array(rs)- np.array(ls),3)
            diff_pxl_min =  diff_pxl.min()
            diff_pxl_max =  diff_pxl.max()
            diff_mtr = np.round((np.array(rs)- np.array(ls))*self.LeftLane.MX,3)
            diff_mtr_min =  diff_mtr.min()
            diff_mtr_max =  diff_mtr.max()
            print('        Y : {}'.format(self.LeftLane.y_checkpoints))
            print(' Left   X : {}  Min: {:8.3f}  Max: {:8.3f}  spread{:8.3f}'.format(ls, ls_min, ls_max, ls_max - ls_min))
            print(' Right  X : {}  Min: {:8.3f}  Max: {:8.3f}  spread{:8.3f}'.format(rs, rs_min, rs_max, rs_max - rs_min))
            print('\n diff pxl : {}  Min: {:8.3f}  Max: {:8.3f}  spread{:8.3f}'.format(diff_pxl, diff_pxl_min, diff_pxl_max, diff_pxl_max - diff_pxl_min))
            print(' diff mtr : {}  Min: {:8.3f}  Max: {:8.3f}  spread{:8.3f}'.format(diff_mtr, diff_mtr_min, diff_mtr_max, diff_mtr_max - diff_mtr_min))
        
        print()
        print('-'*40)
        print('Line Base History:')
        print('-'*40)
        print('Linebase History Left  (m): ', [round(i,3) for i in  self.LeftLane.line_base_meters[-10:]])
        print('Linebase History Right (m): ', [round(i,3) for i in  self.RightLane.line_base_meters[-10:]])
        print('Line width History     (m): ', [round(i-j,3) for i,j in zip(self.RightLane.line_base_meters[-10:], 
                                                                        self.LeftLane.line_base_meters[-10:])])
        display_one(self.histogram, size =(8,4) , title = self.frameTitle)


        # ax.plot(np.array(self.imgThrshldHistory), label = 'Normal/Dark')
        # ax.plot(self.diffsSrcDynPoints , label = 'SrcDynDiff')
        # ax.plot(self.WarpedRGBMean     , label = 'Warped RGB Mean')
        # ax.plot(np.array(videoPipeline.RightLane.RSE_history), label = 'RLane RSE')
        # ax.plot(np.array(videoPipeline.LeftLane.RSE_history), label = 'LLane RSE') 
        # ax.plot(videoPipeline.offctr_history, label='Off Center') ### ,  color=SCORE_COLORS[score_key])
        # ax.plot(np.array(Lane.LSE_history), label = 'LSE')
        # ax.plot(np.array(Lane.RSE_threshold_history), label = 'RSE Throld')

        # min_x = min(self.imgUndistStats['RGB'])
        # max_x = max(self.imgUndistStats['RGB'])
        # ax.plot(self.imgPixelRatio, label = 'Pixel Rto')
        # ax.plot(self.UndistRGBMean, label = 'Undist RGB Mean')
        # ax.plot(self.WarpedRGBMean, label = 'Warped RGB Mean')
        # ax.plot(self.diffsSrcDynPoints, label = 'SrcDynDiff')
        # ax.plot(np.array(self.imgAcceptHistory), label = 'Polynom Acpt/Rjct')
        # ax.plot(np.array(self.imgThrshldHistory), label = 'Normal/Dark')

        # ax.plot(self.imgUndistStats['Hue'], color='r', label='Hue'  )   
        # ax.plot(self.imgUndistStats['Lvl'], color='g', label='Level') 
        # ax.plot(self.imgUndistStats['Sat'], color='b', label='Sat'  )   
        # ax.plot(self.imgUndistStats['RGB'] ,color='k', label='Mean')

        # ax.plot(self.imgUndistStats['Red'], color='r', label='Red') 
        # ax.plot(self.imgUndistStats['Grn'], color='g', label='Grn') 
        # ax.plot(self.imgUndistStats['Blu'], color='b', label='Blu') 
        # ax.plot(self.imgUndistStats['RGB'] , label = 'Undist RGB Mean')

        # ax.plot(self.imgWarpedStats['Hue'], color='r', label='Hue (Warped)', linestyle='dashed')
        # ax.plot(self.imgWarpedStats['Lvl'], color='g', label='Lvl (Warped)', linestyle='dashed') 
        # ax.plot(self.imgWarpedStats['Sat'], color='b', label='Sat (Warped)', linestyle='dashed')
        # ax.plot(self.imgWarpedStats['HLS'] ,color='k', label='HLS (Warped)', linestyle='dashed')

        # min_x = min(self.imgUndistStats['RGB'])
        # max_x = max(self.imgUndistStats['RGB'])

        # ax.plot(np.array(self.RightLane.RSE_history), label = 'RLane RSE')
        # ax.plot(self.diffsSrcDynPoints, label = 'SrcDynDiff')
        # ax.plot(self.LeftLane.pixelRatio , color='r', label='Left') 

        # ax.plot(self.imgPixelRatio, label = 'Pixel Rto')
        # ax.plot(self.UndistRGBMean, label = 'Undist RGB Mean')
        # ax.plot(self.WarpedRGBMean, label = 'Warped RGB Mean')
        # ax.plot(self.diffsSrcDynPoints, label = 'SrcDynDiff')
        # ax.plot(np.array(self.imgAcceptHistory), label = 'Polynom Acpt/Rjct')
        # ax.plot(np.array(self.imgThrshldHistory), label = 'Normal/Dark')
