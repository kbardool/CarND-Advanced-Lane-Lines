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
from collections import deque
from classes.line import Line
from classes.plotting import PlotDisplay
from common.utils import (find_lane_pixels  , search_around_poly, 
                          offCenterMsg      , curvatureMsg      , colorLanePixels   , displayPolynomial      , displayRoILines,  
                          displayDetectedRegion   , displayText , displayGuidelines , displayPolySearchRegion, 
                          display_one, display_two, display_multi )
from common.sobel import  apply_thresholds,  apply_perspective_transform, perspectiveTransform, erodeDilateImage
                                                                                
pp = pprint.PrettyPrinter(indent=2, width=100)
print(os.getcwd())


class ALFPipeline(object):
    NAME                     = 'ALFConfig'
    
    def __init__(self, cameraConfig, **kwargs):

        self.camera                   = cameraConfig
        self.height                   = self.camera.height
        self.width                    = self.camera.width
        self.camera_x                 = self.camera.width //2
        self.camera_y                 = self.camera.height

        self.mode                     = kwargs.get('mode'                 ,    1)
        self.POLY_DEGREE              = kwargs.get('poly_degree'          ,    2)
        self.HISTORY                  = kwargs.get('history'              ,    8)
        self.ERODE_DILATE             = kwargs.get('erode_dilate'         , False)        

        
        self.NWINDOWS                 = kwargs.get('nwindows'             ,   30)
        self.HISTOGRAM_WIDTH_RANGE    = kwargs.get('hist_width_range'     ,  600)
        self.HISTOGRAM_DEPTH_RANGE    = kwargs.get('hist_depth_range'     ,  2 * self.height // 3)
        self.WINDOW_SRCH_MRGN         = kwargs.get('window_search_margin' ,   55)
        self.INIT_WINDOW_SRCH_MRGN    = kwargs.get('init_window_search_margin' ,  self.WINDOW_SRCH_MRGN)
        self.MINPIX                   = kwargs.get('minpix'               ,   90)
        self.MAXPIX                   = kwargs.get('maxpix'               , 8000)
 
        self.POLY_SRCH_MRGN           = kwargs.get('poly_search_margin'   ,   45)
        self.PIXEL_THRESHOLD          = kwargs.get('pixel_threshold'      ,  500)
        self.PIXEL_RATIO_THRESHOLD    = kwargs.get('pixel_ratio_threshold',   30)
        self.LANE_RATIO_THRESHOLD     = kwargs.get('lane_ratio_threshold' ,    5)
        self.RSE_THRESHOLD            = kwargs.get('rse_threshold'        ,   80)

        self.YELLOW_DETECTION_LIMIT   = kwargs.get('yello_limit'          ,   25)
        self.RED_DETECTION_LIMIT      = kwargs.get('red_limit'            ,   75)
        self.OFF_CENTER_ROI_THRESHOLD = kwargs.get('offcntr_roi_threshold',   60)
 
        self.debug                    = kwargs.get('debug'                ,False)
        self.debug2                   = kwargs.get('debug2'               ,False)
        self.debug3                   = kwargs.get('debug3'               ,False)
        # self.debug3 = self.debug or self.debug2 or self.debug3
        self.HISTOGRAM_SEARCH_RANGE   = (self.camera_x - self.HISTOGRAM_WIDTH_RANGE, self.camera_x + self.HISTOGRAM_WIDTH_RANGE)

        ## Thresholding Parameters 
        self.HIGH_RGB_THRESHOLD       = kwargs.get('high_rgb_threshold'   ,  180) # 220)
        self.MED_RGB_THRESHOLD        = kwargs.get('med_rgb_threshold'    ,  180) # 175)   ## chgd from 110 2-26-20
        self.LOW_RGB_THRESHOLD        = kwargs.get('low_rgb_threshold'    ,  100) # 175)   ## chgd from 110 2-26-20
        self.VLOW_RGB_THRESHOLD       = kwargs.get('vlow_rgb_threshold'   ,   35) # 175)   ## chgd from 110 2-26-20
        self.HIGH_SAT_THRESHOLD       = kwargs.get('high_sat_threshold'   ,  110) # 150)
        self.LOW_SAT_THRESHOLD        = kwargs.get('low_sat_threshold'    ,   20) #  20)   ## chgd from 110 2-26-20

        self.XHIGH_THRESHOLDING       = kwargs.get('high_thresholding'    , 'cmb_mag_x')
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
        self.imgCondHistory           = []
        self.acceptHistory            = []
        self.SrcAdjustment_history    = [] 
        self.diffsSrcDynPoints        = []
        self.offctr_history           = []
        self.imgPixelRatio            = []
        self.src_points_history       = [] 
        self.HLS_key                  = ['Hue', 'Lvl', 'Sat']
        self.RGB_key                  = ['Red', 'Grn', 'Blu']
        self.imgUndistStats           = self.initImageInfoDict()
        self.imgWarpedStats           = self.initImageInfoDict()

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
    
        self.LeftLane = Line(name =  'Left', history = self.HISTORY, poly_degree = self.POLY_DEGREE, height = self.height, 
                             y_src_top = self.y_src_top, y_src_bot = self.y_src_bot, rse_threshold = self.RSE_THRESHOLD)
        self.RightLane= Line(name = 'Right', history = self.HISTORY, poly_degree = self.POLY_DEGREE, height = self.height, 
                             y_src_top = self.y_src_top, y_src_bot = self.y_src_bot, rse_threshold = self.RSE_THRESHOLD)
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


    def __call__(self, **kwargs ):
        '''
        '''
        self.image                    = self.inVideo.image   
        self.debug                    = kwargs.get('debug' , False)
        self.debug2                   = kwargs.get('debug2', False)
        self.debug3                   = kwargs.get('debug3', False)
        self.debug4                   = kwargs.get('debug4', False)
        # self.debug3 = self.debug or self.debug2 or self.debug3
        self.displayResults           = kwargs.get('displayResults', False)
        self.frameTitle               = self.inVideo.frameTitle
        
        self.mode                     = kwargs.get('mode'                    , self.mode)
        self.slidingWindowBootstrap   = kwargs.get('slidingWindowBootstrap'  , self.slidingWindowBootstrap) 
        # self.skipFrameDetection       = False
        self.validDetections          = False
        
        ###----------------------------------------------------------------------------------------------
        ### PIPELINE 
        ###----------------------------------------------------------------------------------------------
        self.imgUndist = self.camera.undistortImage(self.image)
        self.saveImageStats(self.imgUndist, self.imgUndistStats)

        self.src_points_history.append(self.src_points)

        self.imgWarped, _ , self.Minv = perspectiveTransform(self.imgUndist, self.src_points, self.dst_points, debug = self.debug4)
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
        self.debugInfo_ImageSummaryInfo()

        if self.debug3:
            self.debugInfo_ImageInfo()

        if self.debug:
            self.debugInfo_srcPointsRoI(title= 'Perspective Tx. source points')

        # if self.skipFrameDetection:
        #     self.build_result_image()
        #     self.debugInfo_Final()
        #     return self.resultImage, self.displayInfo

        ###----------------------------------------------------------------------------------------------
        ### Apply thresholding and Warping of thresholded images 
        ###----------------------------------------------------------------------------------------------
        if self.mode == 1:
            self.image_to_threshold = self.imgUndist
        else:
            self.image_to_threshold = self.imgWarped

        outputs = apply_thresholds(self.image_to_threshold, self.thresholdParms)

        warped_outputs = apply_perspective_transform(outputs, self.thresholdStrs, self.src_points, self.dst_points, 
                                                    size = (15,5), debug = self.debug)
        
        self.imgThrshld = outputs[self.thresholdMethod]
        self.post_threshold = warped_outputs[self.thresholdMethod]

        # if self.mode == 1:  ### Warped AFTER thresholding
            # self.post_threshold, _, Minv = perspectiveTransform(self.imgThrshld, self.src_points, self.dst_points, debug = self.debug4)
        # else:               ### Warped BEFORE thresholding
            # self.post_threshold = self.imgThrshld

        
        ###----------------------------------------------------------------------------------------------
        ##  if ERODE_DILATE flag is True, erode/dilate thresholded image
        ###----------------------------------------------------------------------------------------------
        # if self.ERODE_DILATE:
            # self.working_image = erodeDilateImage(self.post_threshold , ksize = 3, iters = 3)
        # else:
        self.working_image = self.post_threshold


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
            self.findLanePixelsRC, self.out_img, self.histogram, imgPixelRatio = find_lane_pixels(self.working_image, 
                                                                                self.LeftLane, self.RightLane, 
                                                                                nwindows        = self.NWINDOWS, 
                                                                                histWidthRange  = self.HISTOGRAM_WIDTH_RANGE, 
                                                                                histDepthRange  = self.HISTOGRAM_DEPTH_RANGE, 
                                                                                window_margin   = window_search_margin, 
                                                                                debug = self.debug) 

        else:    
            self.findLanePixelsRC, self.out_img, self.histogram, imgPixelRatio = search_around_poly(self.working_image, 
                                                                                self.LeftLane, self.RightLane, 
                                                                                search_margin   = self.POLY_SRCH_MRGN, 
                                                                                pixel_thr       = self.PIXEL_THRESHOLD,
                                                                                pixel_ratio_thr = self.PIXEL_RATIO_THRESHOLD,
                                                                                lane_ratio_thr  = self.LANE_RATIO_THRESHOLD,
                                                                                debug = self.debug)
        
        self.imgPixelRatio.append(imgPixelRatio)

        self.assess_lane_detection_results()
        ###----------------------------------------------------------------------------------------------
        ### Fit polynomial on found lane pixels 
        ###----------------------------------------------------------------------------------------------
        self.fit_polynomial_process_v2()
    

        if self.displayResults:
            self.displayFittingInfo()            

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
        self.debugInfo_Final()

        
        if self.firstFrame :
            self.firstFrame = False

        return self.resultImage, self.displayInfo
            





    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def assess_lane_detection_results(self):

        # if  self.slidingWindowBootstrap or self.RoIAdjustment:
            # self.fitPolynomials = True
            # return 

        self.fitPolynomials = self.findLanePixelsRC

        if self.debug:
            print()
            print('assess_lane_detection_results()')
            print('-'*20)
            print(' findLanePixels RC: ' , self.findLanePixelsRC)

        for Lane in [self.LeftLane, self.RightLane]:
            
            if (Lane.pixelCount < self.PIXEL_THRESHOLD): 
                print()
                print('-'*100)
                print(' {} Lane pixels count under threshold '.format(Lane.name))
                print(' {} PixelCount: {:7.2f}  <  Count Min Threshold: ({:4d}) '.format(
                        Lane.name, Lane.pixelCount, self.PIXEL_THRESHOLD))
                print('-'*100)
                Lane.fitPolynomials = False

            elif (Lane.pixelRatio[-1]  < self.LANE_RATIO_THRESHOLD): 
                print()
                print('-'*100)
                print(' {} Lane pixel Ratio under Threshold '.format(Lane.name))
                print(' {} Pxl Ratio: {:7.2f} < Lane Threshold: ({:4d})'.format(Lane.name, Lane.pixelRatio[-1], self.LANE_RATIO_THRESHOLD))
                print(' {} Pxl Count: {:7.2f} - Count Threshold: ({:4d})'.format(Lane.name,Lane.pixelCount, self.PIXEL_THRESHOLD))
                print('-'*100)
                Lane.fitPolynomials = False
            else:
                Lane.fitPolynomials = True

        # if (self.imgPixelRatio[-1]< self.PIXEL_RATIO_THRESHOLD) and (self.imgWarpedStats['Sat'][-1] > self.HIGH_SAT_THRESHOLD):
        if (self.imgWarpedStats['RGB'][-1]> self.HIGH_RGB_THRESHOLD) and (self.imgWarpedStats['Sat'][-1] > self.HIGH_SAT_THRESHOLD):
            print('-'*100)
            print(' High Mean RGB and Saturation on Warped Image: ')
            print(' imgPixelRatio {:7.2f}  > PIXEL_RATIO_THRESHOLD ({:7.2f})  AND  imgWarpedStats[Sat] {:7.2f}  > HIGH_SAT_THRESHOLD ({:7.2f})'.
            format(self.imgWarpedStats['RGB'][-1], self.HIGH_RGB_THRESHOLD,self.imgWarpedStats['Sat'][-1] , self.HIGH_SAT_THRESHOLD))
            print(' left Pxl Ratio: {:7.2f}  or  right Pxl Ratio: {:7.2f}  - Lane Threshold: ({:4d})     imgPixelRatio: {:7.2f} - ImagePxl Thr:({:4d})'.
                     format( self.LeftLane.pixelRatio[-1], self.RightLane.pixelRatio[-1], self.LANE_RATIO_THRESHOLD, 
                            self.imgPixelRatio[-1], self.PIXEL_RATIO_THRESHOLD))
            print(' leftPixelCount: {:7.2f}  or  rightPixelCount: {:7.2f}  -  Count Min Threshold: ({:4d}) '.format(
                      self.LeftLane.pixelCount, self.RightLane.pixelCount, self.PIXEL_THRESHOLD))
            print('-'*100)
            imageFitPolynomials = False
        else:
            imageFitPolynomials = True


        self.fitPolynomials = self.LeftLane.fitPolynomials and  self.RightLane.fitPolynomials and imageFitPolynomials
        if self.debug:
            print(' Lane fitPolynomials :   Left: {}   Right:{}  Image fitPolynomials: {}    final fitPolynomials: {} '.format(
                self.LeftLane.fitPolynomials, self.RightLane.fitPolynomials, imageFitPolynomials, self.fitPolynomials))

    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def fit_polynomial_process_v2(self):
        ## If lane pixel detection was successful, try to fit polynomials over detected pixels 

        for Lane in [self.LeftLane, self.RightLane]:

            if  Lane.fitPolynomials:

                Lane.fit_polynomial(debug  = self.debug)
                Lane.poly_assess_result = Lane.assessFittedPolynomial(debug = self.debug)

                ## Realignment of the perspective transformation window will reuslt in a 
                ## High RSE Error. We will allow this error rate when it is a result of a 
                ## RoI realignment. Other wise proceed nornally.
                if  (self.slidingWindowBootstrap and self.RoIAdjustment):
                    msg = ' RoIAdjustment performed - Polynomial fit will be accepted \n'
                    Lane.acceptPolynomial = True
                    # self.slidingWindowBootstrap  = False
                    Lane.reset_best_fit(debug = self.debug)
                else:
                    Lane.acceptPolynomial = Lane.poly_assess_result 
                    msg = ' {} Lane acceptPolynomial = {}'.format(Lane.name, Lane.acceptPolynomial)

            else:
                msg = ' {} lane detection failed on frame: {} '.format(Lane.name, self.frameTitle)
                Lane.acceptPolynomial = False

            if self.debug :
                print(msg)        

            if Lane.acceptPolynomial:
                Lane.acceptFittedPolynomial(debug = self.debug, debug2 = self.debug2)
            else:
                Lane.rejectFittedPolynomial(debug = self.debug, debug2 = self.debug2)


        ### Frame level actions that need to be taken based on
        ### acceptance or rejection of polynomials 

        self.acceptPolynomial = self.LeftLane.acceptPolynomial and self.RightLane.acceptPolynomial
        
        if self.debug:
            print('\n acceptPolynomial = poly_left ({}) and poly_right({}) = {}'.format(
                   self.LeftLane.acceptPolynomial, self.RightLane.acceptPolynomial, self.acceptPolynomial)) 

        if self.acceptPolynomial:
            self.ttlAcceptedFrames += 1
            self.ttlRejectedFramesSinceAccepted = 0
            self.ttlAcceptedFramesSinceRejected += 1
            self.validDetections   = True
            self.polyRegionColor1 = 'green'
            self.acceptHistory.append(0)
            self.slidingWindowBootstrap  = False
        else:
            self.ttlRejectedFrames += 1
            self.ttlAcceptedFramesSinceRejected = 0 
            self.ttlRejectedFramesSinceAccepted += 1
            self.slidingWindowBootstrap  = True       
            self.validDetections  = True
            self.polyRegionColor1 = 'yellow' 
            self.acceptHistory.append(-10)

            if self.ttlRejectedFramesSinceAccepted > self.YELLOW_DETECTION_LIMIT:
                self.slidingWindowBootstrap  = True        
                self.polyRegionColor1 = 'salmon'       
                self.acceptHistory.append(-10)
            
            if self.ttlRejectedFramesSinceAccepted > self.RED_DETECTION_LIMIT:
                self.slidingWindowBootstrap  = True
                self.validDetections  = False
                self.polyRegionColor1 = 'red'       
                self.acceptHistory.append(-20)

        ### Display debug info
        if self.debug: 

            if self.acceptPolynomial:
                print('\n => ACCEPT propsed polynomials for frame {:5d}  Accepted frames Since Last Rejected - Left: {:3d}  Right: {:3d} \n'.format(
                    self.inVideo.currFrameNum, self.LeftLane.ttlAcceptedFramesSinceRejected, self.RightLane.ttlAcceptedFramesSinceRejected))
            else:

                print('\n => REJECT proposed polynomials for frame {:5d}  Rejected frames Since Last Detected - Left: {:3d}  Right: {:3d} \n'.format(  
                    self.inVideo.currFrameNum, self.LeftLane.ttlRejectedFramesSinceDetected, self.RightLane.ttlRejectedFramesSinceDetected))

            self.debugInfo_DetectionInfo()



    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def build_result_image(self, **kwargs):
        disp_start = kwargs.get('start' , self.displayRegionTop)
        disp_end   = kwargs.get('end'   , self.displayRegionBot)
        beta       = kwargs.get('beta'  , 0.5)
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
        
        # if self.validDetections:
        self.resultImage, self.dyn_src_points_list = displayDetectedRegion(self.imgUndist, 
                                                                     self.LeftLane.fitted_best , 
                                                                     self.RightLane.fitted_best, 
                                                                     self.Minv, 
                                                                     disp_start = disp_start, 
                                                                     disp_end   = disp_end  ,
                                                                     beta  = beta , 
                                                                     color = self.polyRegionColor1, 
                                                                     frameTitle = self.frameTitle, 
                                                                     debug = self.debug2)
        # else:
            # self.resultImage = np.copy(self.imgUndist)

        displayText(self.resultImage, 40, 40, self.frameTitle, fontHeight = 20)
        if self.validDetections:
            displayText(self.resultImage, 40, 80, self.curv_msg  , fontHeight = 20)
            displayText(self.resultImage, 40,120, self.oc_msg    , fontHeight = 20)
        else:
            displayText(self.resultImage, 40, 80, 'Unable to detect lanes' , fontHeight = 20)

        # displayGuidelines(self.resultImage, draw = 'y');
        return


    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def adjust_RoI_window(self, **kwargs):
        '''
        Adjust the perspective tranformation source / dest points based on predefined criteria
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
            print('adjust_RoI_window() - FirstFrame:', self.firstFrame, ' AcceptPolynomial:', self.acceptPolynomial )
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


        ###----------------------------------------------------------------------------------------------
        # if quality of last image threshold is > %80 and we need to run a bootstrap, set up to do so in
        # next video frame
        ###----------------------------------------------------------------------------------------------
        if  (self.acceptPolynomial) and \
            ( ( max_diffs > self.OFF_CENTER_ROI_THRESHOLD ) or (self.firstFrame)):
            # print('  ','-'*95)
            print()
            print('    Adjust perspective transform source points -  OffCtr Pxls: {}    max_diffs: {}    imgPxlRatio: {} '.format(off_center_pixels, 
                    max_diffs, self.imgPixelRatio[0]))
            print('   ','-'*100)
            print('    Cur src_points_list :  {} '.format(self.src_points_list))
            print()
            print('    New src_points_list :  {} '.format(self.dyn_src_points_list))
            print('       Prev Left x_base : ', self.LeftLane.x_base[-2], '   Right x_base  :', self.RightLane.x_base[-2])
            print('       New  Left x_base : ', self.LeftLane.x_base[-1], '   Right x_base  :', self.RightLane.x_base[-1])
            print()

            self.debugInfo_srcPointsRoI(title= 'source points prior to realignment')
            self.debugInfo_newSrcPointsRoI(title= 'source points after realignment')


            self.prev_src_points_list = self.src_points_list
            self.src_points_list      = self.dyn_src_points_list
            self.src_points           = np.array(self.dyn_src_points_list, dtype = np.float32)

            self.slidingWindowBootstrap  = True
            self.RoIAdjustment           = True
            self.SrcAdjustment_history.append((len(self.offctr_history), self.offctr_history[-1], self.diffsSrcDynPoints[-1]))
            self.LeftLane.x_base.append( self.dyn_src_points_list[3][0])
            self.RightLane.x_base.append(self.dyn_src_points_list[2][0])

        else:
            self.RoIAdjustment = False

        return 

    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def debugInfo_ImageSummaryInfo(self):
        print('{:25s}- UNDIST  RGB: {:3.0f}  HLS:{:3.0f} -({:3.0f},{:3.0f},{:3.0f})'\
                '    WARPED RGB: {:3.0f}  HLS: {:3.0f}  Hue:{:3.0f}  Lvl: {:3.0f}  SAT: {:3.0f}'\
                '    {:5s} - {:10s}  '.format(self.frameTitle,
                self.imgUndistStats['RGB'][-1], self.imgUndistStats['HLS'][-1], 
                self.imgUndistStats['Hue'][-1], self.imgUndistStats['Lvl'][-1], self.imgUndistStats['Sat'][-1],
                self.imgWarpedStats['RGB'][-1], self.imgWarpedStats['HLS'][-1], 
                self.imgWarpedStats['Hue'][-1], self.imgWarpedStats['Lvl'][-1], self.imgWarpedStats['Sat'][-1],     
                self.Conditions.upper(), self.thresholdMethod))
        if self.debug:
            print( ' Thresholds:  HIGH RGB: {}    MED RGB: {}   LOW RGB: {}  VLOW RGB: {}   HIGH SAT: {}   LOW SAT: {} '.
                format(self.HIGH_RGB_THRESHOLD, self.MED_RGB_THRESHOLD , self.LOW_RGB_THRESHOLD, 
                        self.VLOW_RGB_THRESHOLD, self.HIGH_SAT_THRESHOLD, self.LOW_SAT_THRESHOLD))                


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
    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def debugInfo_ThresholdedImage(self):
        if self.ERODE_DILATE:
            display_two(self.imgThrshld, self.post_threshold, title1 = self.thresholdMethod +' '+str(np.sum(self.imgThrshld)), 
                                                            title2 = 'post_threshold (pre erode/dilate)- '+str(np.sum(self.post_threshold)))

        display_two(self.imgThrshld, self.working_image, title1 = self.thresholdMethod +' '+str(np.sum(self.imgThrshld)), 
                                                            title2 = 'post_threshold - '+str(np.sum(self.working_image)))

    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def debugInfo_DetectionInfo(self, display = True):
                    
        prevBestFit  = colorLanePixels(self.out_img, self.LeftLane, self.RightLane)

        if self.HISTORY > 1:
            prevBestFit = displayPolynomial(prevBestFit, self.LeftLane.fitted_best_history, self.RightLane.fitted_best_history, iteration = -2, 
                                            color = 'aqua')
            prevBestFit = displayPolynomial(prevBestFit, self.LeftLane.fitted_current, self.RightLane.fitted_current, iteration = -1, color = 'yellow')

        currentFit  = displayPolynomial(prevBestFit, self.LeftLane.fitted_current, self.RightLane.fitted_current, iteration = -1, color = 'yellow')

        currentFit = displayPolynomial(currentFit, self.LeftLane.fitted_best, self.RightLane.fitted_best, color = 'red')
        if display:
            print(' x_src_top_left : {}   x_src_top_right: {}   x_src_bot_left: {}   x_src_bot_right: {}'.format(
                self.src_points_list[0], self.src_points_list[1],self.src_points_list[3],self.src_points_list[2]))
            # print(' y_src_top_left : {}   y_src_top_right: {}   y_src_bot_left: {}   y_src_bot_right: {}'.format(self.dst_points_list))
                # self.y_src_top, self.y_src_top, self.y_src_bot, self.y_src_bot))  
            display_two(self.working_image, self.out_img, title1 = 'working_image - '+self.frameTitle, 
                                                          title2 = 'out_img ')
            display_two(prevBestFit, currentFit  , title1 = 'Prev best fit (Black: Prev fit, Yellow: New proposal)' , 
                                                   title2 = 'ImgLanePxls (Black: Prev fit, Yellow: New fit, Red: Best Fit)' )
            print()
        
        return currentFit

    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def debugInfo_srcPointsRoI(self, size = (24,9), title = None ):
        print()
        print('     x_top_disp : {:<13d}     x_src_center : {:<13d}      x_bot_disp : {:<4d} '.format(self.x_top_disp, self.x_src_center, self.x_bot_disp))
        print(' x_src_top_left : {:12s}   x_src_top_right : {:12s}   x_src_bot_left : {:12s}   x_src_bot_right : {:12s}'.format(
                str(self.src_points_list[0]), str(self.src_points_list[1]), str(self.src_points_list[3]), str(self.src_points_list[2])))
        print(' y_src_top_left : {:12s}   y_src_top_right : {:12s}   y_src_bot_left : {:12s}   y_src_bot_right : {:12s}'.format(
                str(self.dst_points_list[0]), str(self.dst_points_list[1]), str(self.dst_points_list[3]), str(self.dst_points_list[2])))  
        display_two(self.imgRoI  , self.imgRoIWarped, title1 = title , grid1 = 'minor',
                                            title2 = title + ' - warped', grid2 = 'major', size = size)
        print()

        return  

    def debugInfo_newSrcPointsRoI(self, display = True, size = (24,9), title = None):

        imgRoI             = displayRoILines(self.imgUndist, self.dyn_src_points_list , color = 'blue', thickness = 2)
        imgRoIWarped, _, _ = perspectiveTransform(imgRoI   , self.dyn_src_points      , self.dst_points)
        imgRoIWarped       = displayRoILines(imgRoIWarped  , self.dst_points_list     , thickness = 2, color = 'yellow')
        
        display_two(imgRoI  , imgRoIWarped  , title1 = title , grid1 = 'minor',
                                              title2 = title+' - Warped' , grid2 = 'major', size = size)
        
        return imgRoI, imgRoIWarped 

    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def debugInfo_RoITransforms(self):
        self.debugInfo_srcPointsRoI(title= 'source points prior to realignment')
        self.debugInfo_newSrcPointsRoI()

    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def debugInfo_Final(self, **kwargs):
        if (not self.displayResults):
            self.displayInfo = None
        else:

            debug   = kwargs.get('debug', False)
            debug2  = kwargs.get('debug2', False)
            debug3  = kwargs.get('debug3', False)
            debug4  = kwargs.get('debug4', False)
            polyRegionColor1 = kwargs.get('color1', 'green')

            if debug:
                print(' Left lane MR fit           : ', self.LeftLane.curr_fit , '    Right lane MR fit     : ', self.RightLane.curr_fit)
                print(' Left lane MR best fit      : ', self.LeftLane.best_fit    , '    Right lane MR best fit: ', self.RightLane.best_fit)
                print(' Left radius @ y =  10   : '+str(self.LeftLane.get_radius(10)) +" m   Right radius: "+str(self.RightLane.get_radius(10))+" m")
                print(' Left radius @ y = 700   : '+str(self.LeftLane.get_radius(700))+" m   Right radius: "+str(self.RightLane.get_radius(700))+" m")
                print(' Curvature message : ', curv_msg)
                print(' Off Center Message: ', oc_msg)            

            result_1, _  = displayDetectedRegion(self.imgUndist, self.LeftLane.fitted_current, self.RightLane.fitted_current, 
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
            imgLanePxls = self.debugInfo_DetectionInfo(display = False)

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
                imgThrshldWarped = self.working_image
            else: 
                # print(' Display mode 2')
                ### results of applying thresholding BEFORE warping undistorted image 
                thresholdParms   = self.ImageThresholds[1][self.Conditions] 
                output2          = apply_thresholds(self.imgUndist, thresholdParms, debug = debug2)  
                self.imgThrshld  = output2[self.thresholdMethod]
                self.imgThrshldWarped, _, _  = perspectiveTransform(self.imgThrshld, self.src_points, self.dst_points, debug = debug4) 
                imgWarpedThrshld = self.working_image
                
            self.displayInfo = PlotDisplay(6,2)
            self.displayInfo.addPlot(self.image       , title = 'original frame - '+self.frameTitle)
            self.displayInfo.addPlot(self.imgUndist   , title = 'imgUndist - Undistorted Image')
            
            self.displayInfo.addPlot(self.imgRoI      , title = 'imgRoI'   )
            self.displayInfo.addPlot(self.imgRoIWarped, title = 'imgRoIWarped' )
            
            self.displayInfo.addPlot(self.imgThrshld  , title = 'imgThrshld - Thresholded image')
            self.displayInfo.addPlot(self.imgWarped   , title = 'imgWarped - Warped Image')
            
            self.displayInfo.addPlot(imgThrshldWarped, title = 'imgThrshldWarped - Img Thresholded ---> Warped (Mode 1)')
            self.displayInfo.addPlot(self.imgWarpedThrshld, title = 'imgWarpedThrshld - Img Warped ---> Thresholded (Mode 2)')
            
            self.displayInfo.addPlot(imgLanePxls      , title = 'ImgLanePxls (Black: Prev fit, Yellow: New fit, Red: Best Fit)' )
            self.displayInfo.addPlot(self.histogram   , title = 'Histogram of activated pixels', type = 'plot' )
            
            self.displayInfo.addPlot(result_1         , title = 'result_1 : Using LAST fit')
            self.displayInfo.addPlot(self.resultImage , title = 'finalImage : Using BEST fit'+self.frameTitle)
            self.displayInfo.closePlot()
        return 
    

    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def displayFittingInfo(self):
        np_format = {}
        np_format['float'] = lambda x: "%8.2f" % x
        np_format['int']   = lambda x: "%8d" % x
        np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =100, formatter = np_format)
        
        print()
        print('='*70)
        print('Display fitting info for ', self.frameTitle)
        print('='*70)

        # self.debugInfo_DetectionInfo()
        print()
        print('Proposed Polynomial:')
        print('-'*40)
        print('left      : {}    right     : {} '.format(self.LeftLane.curr_fit, self.RightLane.curr_fit))

        if len(self.LeftLane.curr_fit_history) > 0:
            print()
            print('Current Best Fit Polynomial:')
            print('-'*40)
            print('left      : {}    right     : {} '.format(self.LeftLane.best_fit, self.RightLane.best_fit))

            print()
            print('Diff b/w proposed and best_fit polynomial ')
            print('-'*40)
            print('left      : {}    right     : {} '.format( self.LeftLane.best_fit-self.LeftLane.curr_fit, 
                                                            self.RightLane.best_fit-self.RightLane.curr_fit)    )
            print()
            print('Proposed RSE with best fit - self.LeftLane: {}   RLane :  {} '.format(self.LeftLane.RSE ,self.RightLane.RSE ))
            print()
            # print('Best RSE Hist LLane : ',  self.LeftLane.RSE_history[-15:])
            # print('Best RSE Hist RLane : ', self.RightLane.RSE_history[-15:])
            # print('Best fit RSE Hist LeftLane  : ', ['{:8.3f}'.format(i) for i in self.LeftLane.RSE_history])
            # print('Best fit RSE Hist RightLane : ', ['{:8.3f}'.format(i) for i in self.RightLane.RSE_history])
            
        print()
        print('-'*40)
        print('Previously Accepted Fit Polynomials:')
        print('-'*40)
        i = 0
        for ls, rs in zip(reversed(self.LeftLane.curr_fit_history), reversed(self.RightLane.curr_fit_history)):
            i -= 1
            print('left[{:2d}]    : {}    right[{:2d}]     : {} '.format(i,ls, i,rs))
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
        print('Lane Radius from latest proposed fit:')
        print('-'*40)
        ls = self.LeftLane.current_radius
        rs = self.RightLane.current_radius
        diff = np.round(np.array(rs)- np.array(ls),3)
        avg  = np.round((np.array(rs) + np.array(ls))/2,3)
        print('Left   Y   : ', self.LeftLane.y_checkpoints)
        print('left       : ',ls, '  Avg:', np.round(np.mean(ls),3))
        print('Right  Y   : ', self.RightLane.y_checkpoints)
        print('right      : ',rs, '  Avg:', np.round(np.mean(rs),3))
        print()
        print('avg        : ',avg)
        print('diff       : ',diff, '  Max:', diff[0:5].max())

        if (self.HISTORY > 1)  and  len(self.LeftLane.best_fit_history) > 0:
            print()
            print('Lane Radius from BEST line fit:')
            print('-'*40)
            ls =  self.LeftLane.best_radius
            rs = self.RightLane.best_radius
            diff = np.round(np.array(rs)- np.array(ls),3)
            avg  = np.round((np.array(rs) + np.array(ls))/2,3)
            print('Left   Y  : ', self.LeftLane.y_checkpoints)
            print('left      : ', ls, '  Avg:', np.round(np.mean(ls),3))
            print('Right  Y  : ', self.RightLane.y_checkpoints)
            print('right     : ', rs, '  Avg:', np.round(np.mean(rs),3))
            print()
            print('avg       : ', avg)
            print('diff      : ', diff, '  Max:', diff[0:5].max())
        
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
        print('Left   Y  : ', self.LeftLane.y_checkpoints)
        print('left      : ',ls  , ' Avg:', np.round(np.mean(ls),3))
        print('Right  Y  : ', self.RightLane.y_checkpoints)
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
            print('Left   Y  : ', self.LeftLane.y_checkpoints)
            print('left      : ',ls  , ' Avg:', np.round(np.mean(ls),3))
            print('Right  Y  : ', self.RightLane.y_checkpoints)
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
        print('Line X Position from latest proposed fit:')
        print('-'*40)
        ls = self.LeftLane.current_linepos
        rs = self.RightLane.current_linepos
        diff_pxl = np.round( np.array(rs)- np.array(ls),3)
        diff_mtr = np.round((np.array(rs)- np.array(ls))*self.LeftLane.MX,3)
        print('- Curr_linepos')
        print('Left   Y : ', self.LeftLane.y_checkpoints)
        print('left   X : ',ls, ' Min:', np.round(ls.min(),3) , ' Max: ',np.round(ls.max(),3))
        print('Right  Y : ', self.RightLane.y_checkpoints)
        print('right  X : ',rs, ' Min:', np.round(rs.min(),3) , ' Max: ',np.round(rs.max(),3))
        print()
        print('diff pxl : ', diff_pxl)
        print('diff mtr : ', diff_mtr, ' Min: {}  Max: {}'.format( diff_mtr[0:5].min(), diff_mtr[0:5].max()))


        if len(self.LeftLane.best_fit_history) > 0:
            print()
            print('Line X Position from best fit:')
            print('-'*40)
            ls = self.LeftLane.best_linepos
            rs = self.RightLane.best_linepos
            diff_pxl = np.round( np.array(rs)- np.array(ls),3)
            diff_mtr = np.round((np.array(rs)- np.array(ls))*self.LeftLane.MX,3)
            print('left   Y : ', self.LeftLane.y_checkpoints)
            print('left   X : ',ls , ' Min:', np.round(ls.min(),3) , ' Max: ',np.round(ls.max(),3))
            print('right  Y : ', self.LeftLane.y_checkpoints)
            print('right  X : ',rs , ' Min:', np.round(rs.min(),3) , ' Max: ',np.round(rs.max(),3))
            print()
            print('diff pxl : ', diff_pxl)
            print('diff mtr : ', diff_mtr, ' Min: {}  Max: {}'.format( diff_mtr[0:5].min(), diff_mtr[0:5].max()))
        
        print()
        print('-'*40)
        print('Line Base History:')
        print('-'*40)
        print('Linebase History Left  (m): ', [round(i,3) for i in  self.LeftLane.line_base_meters[-10:]])
        print('Linebase History Right (m): ', [round(i,3) for i in  self.RightLane.line_base_meters[-10:]])
        print('Line width History     (m): ', [round(i-j,3) for i,j in zip(self.RightLane.line_base_meters[-10:], 
                                                                                    self.LeftLane.line_base_meters[-10:])])
        display_one(self.histogram, size =(8,4) , title = self.frameTitle)



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


    def plot_1(self, legend = 'best', size=(15,7)):
        plt.figure(figsize=size)
        ax = plt.gca()
        min_x = min(self.imgUndistStats['RGB'])
        max_x = max(self.imgUndistStats['RGB'])
        len_x = len(self.imgUndistStats['RGB'])
        # ax.plot(self.offctr_history    , label = 'Off Center') ### ,  color=SCORE_COLORS[score_key])
        ax.plot(self.imgPixelRatio     , label = 'Pixel Rto')
        ax.plot(self.imgUndistStats['RGB'] , label = 'Undist RGB Mean')
   
        ax.plot(np.array(self.acceptHistory), label = 'Polynom Acpt/Rjct')
        ax.plot(np.array(self.imgCondHistory), label = 'ImgCondition')
        ax.set_title('Plot 1 - Pxl Thrshld: {:3d} OffCt Thrshld: {:3d}'.format( self.PIXEL_RATIO_THRESHOLD, self.OFF_CENTER_ROI_THRESHOLD ))
        plt.hlines( self.HIGH_RGB_THRESHOLD      , 0, len_x, color='red' , alpha=0.5, linestyles='dashed', linewidth=1, label = 'HI RGB')    
        plt.hlines( self.MED_RGB_THRESHOLD       , 0, len_x, color='green' , alpha=0.5, linestyles='dashed', linewidth=1, label = 'MED RGB')    
        plt.hlines( self.LOW_RGB_THRESHOLD       , 0, len_x, color='yellow' , alpha=0.5, linestyles='dashed', linewidth=1, label = 'LOW RGB')    
        plt.hlines( self.PIXEL_RATIO_THRESHOLD   , 0, len_x, color='red'  , alpha=0.5, linestyles='dashed', linewidth=1, label='PxlRatioThr')    
        plt.hlines( self.OFF_CENTER_ROI_THRESHOLD, 0, len_x, color='black', alpha=0.8, linestyles='dashed', linewidth=1, label='OffCtrRoIThr' )    
        plt.hlines(-self.OFF_CENTER_ROI_THRESHOLD, 0, len_x, color='black', alpha=0.8, linestyles='dashed', linewidth=1)    
        
        for (x,_,_) in self.SrcAdjustment_history:
            ax.plot(x,-40, 'bo') 
        leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
        leg.set_title('Legend',prop={'size':11})


    def plot_2(self, Lane, ttl = '' ,legend = 'best', size=(15,7)):
        plt.figure(figsize=size)
        ax = plt.gca()
        min_x = min(self.imgUndistStats['RGB'])
        max_x = max(Lane.RSE_history)
        len_x = len(self.imgUndistStats['RGB'])
        ax.plot(Lane.pixelRatio, label = 'Pixel Rto')
        ax.plot(np.array(Lane.RSE_history), label = 'RSE Error')
        ax.plot(np.array(self.acceptHistory), label = 'Polynom Acpt/Rjct')
        

        plt.hlines( 80 , 0, len_x, color='green' , alpha=0.8, linestyles='dashed', linewidth=1, label = '<80>')    
        plt.hlines( 50 , 0, len_x, color='blue'  , alpha=0.8, linestyles='dashed', linewidth=1, label = '<50>')    
        plt.hlines( 15 , 0, len_x, color='maroon', alpha=0.8, linestyles='dashed', linewidth=1, label = '<15>')    
        # plt.hlines( self.OFF_CENTER_ROI_THRESHOLD, 0, len_x, color='black', alpha=1.0, linestyles='dashed', linewidth=1)    
        # plt.hlines(-self.OFF_CENTER_ROI_THRESHOLD, 0, len_x, color='black', alpha=1.0, linestyles='dashed', linewidth=1)    
        
        for (x,_,_) in self.SrcAdjustment_history:
            ax.plot(x,-20, 'bo') 
        plt.ylim(-25, max_x + 5)

        ax.set_title(' Plot 2 - '+Lane.name+ ' Lane - Pixel Threshld: {:3d}       OffCtr Thrshld: {:3d}'.format( 
                     self.PIXEL_RATIO_THRESHOLD, self.OFF_CENTER_ROI_THRESHOLD ))
        leg = plt.legend(loc=legend,frameon=True, fontsize = 12,markerscale = 6)
        leg.set_title('Legend',prop={'size':11})
                                        


    def plot_3(self, legend = 'best', size=(15,7)):
        plt.figure(figsize=size)
        ax = plt.gca()
        len_x = len(self.imgUndistStats['RGB'])

        ax.plot(self.imgWarpedStats['Hue'], color='r', label='Hue (W)')
        ax.plot(self.imgWarpedStats['Sat'], color='b', label='Sat (Warped)')
        ax.plot(self.imgWarpedStats['Lvl'], color='g', label='Lvl (Warped')
        ax.plot(self.imgWarpedStats['RGB'], color='darkorange', label='RGB (Warped)')

        ax.plot(self.imgUndistStats['Sat'], color='darkblue', alpha = 0.5, label='Sat')
        ax.plot(self.imgUndistStats['RGB'], color='darkorange', alpha = 0.5,   label='RGB ')

        # ax.plot(self.imgUndistStats['Red'], color='r', label='Red ')
        # ax.plot(self.imgUndistStats['Grn'], color='g', label='Grn ')
        # ax.plot(self.imgUndistStats['Blu'], color='b', label='Blu ')
        # ax.plot(self.imgUndistStats['Hue'], color='r', label='Hue')
        # ax.plot(self.imgWarpedStats['Red'], color='r', label='Red (Warped)', linestyle='dashed')
        # ax.plot(self.imgWarpedStats['Grn'], color='g', label='Grn (Warped)', linestyle='dashed')
        # ax.plot(self.imgWarpedStats['Blu'], color='b', label='Blu (Warped)', linestyle='dashed')
        # ax.plot(self.imgWarpedStats['Hue'], color='r', label='Hue (Warped)', linestyle='dashed')
        
        ax.plot(np.array(self.imgCondHistory), label = 'Normal/Dark')
        ax.plot(self.imgPixelRatio, color='sienna', label = 'Pixel Rto')

        # plt.hlines( 180      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = '(180)')    
        # plt.hlines( 150      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = '(150)')    
        plt.hlines( self.HIGH_RGB_THRESHOLD    , 0, len_x, color='darkred', alpha=0.5, linestyles='dashed', linewidth=1, 
                    label = 'High RGB '+str(self.HIGH_RGB_THRESHOLD))    
        plt.hlines( self.MED_RGB_THRESHOLD    , 0, len_x, color='darkorange', alpha=0.7, linestyles='dashed', linewidth=1, 
                    label = 'Med RGB '+str(self.MED_RGB_THRESHOLD))    
        plt.hlines( self.LOW_RGB_THRESHOLD    , 0, len_x, color='darkgreen' , alpha=0.5, linestyles='dashed', linewidth=1,
                    label = 'Low RGB '+str(self.LOW_RGB_THRESHOLD))    
        plt.hlines( self.PIXEL_RATIO_THRESHOLD, 0, len_x, color='red'  , alpha=0.5, linestyles='dashed', linewidth=1)    
        ax.set_title('Plot 3 - Image RGB Avgs - Pxl Thrshld: {:3d} OffCt Thrshld: {:3d}'.format( self.PIXEL_RATIO_THRESHOLD, self.OFF_CENTER_ROI_THRESHOLD ))

        for (x,_,_) in self.SrcAdjustment_history:
            ax.plot(x,-20, 'bo') 
        leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
        leg.set_title('Legend',prop={'size':11})

    def plot_4U(self, legend = 'best', size=(15,7), pxlthr=False):
        plt.figure(figsize=size)
        ax = plt.gca()
        len_x = len(self.imgUndistStats['Hue'])

        ax.plot(self.imgUndistStats['Hue'], color='r', label='Hue')
        ax.plot(self.imgUndistStats['Lvl'], color='g', label='Lvl')
        ax.plot(self.imgUndistStats['Sat'], color='b', label='Sat')
        # ax.plot(self.imgUndistStats['HLS'] ,color='k', label='HLS')
        ax.plot(self.imgUndistStats['RGB'] ,color='darkorange', label='RGB')
        
        ax.plot(np.array(self.imgCondHistory), label = 'ImgCondition')
        # ax.plot(np.array(self.acceptHistory) , label = 'Polynom Acpt/Rjct')
        if pxlthr:
            ax.plot(self.imgPixelRatio, color='sienna', label = 'Pixel Rto')

        plt.hlines( self.HIGH_RGB_THRESHOLD    , 0, len_x, color='darkred', alpha=0.5, linestyles='dashed', linewidth=1, 
                    label = 'High RGB '+str(self.HIGH_RGB_THRESHOLD))    
        plt.hlines( self.MED_RGB_THRESHOLD , 0, len_x, color='darkgreen', alpha=0.5, linestyles='dashed', linewidth=1, 
                    label = 'Med RGB '+str(self.MED_RGB_THRESHOLD))    
        plt.hlines( self.LOW_RGB_THRESHOLD , 0, len_x, color='brown', alpha=0.5, linestyles='dashed', linewidth=1,  
                    label = 'Low RGB '+str(self.LOW_RGB_THRESHOLD))    
        plt.hlines( self.VLOW_RGB_THRESHOLD, 0, len_x, color='red'   , alpha=0.5, linestyles='dashed', linewidth=1,  
                    label = 'VLow RGB '+str(self.VLOW_RGB_THRESHOLD))    
        # plt.hlines( self.LOW_SAT_THRESHOLD , 0, len_x, color='g'   , alpha=0.5, linestyles='dashed', linewidth=1,
        #             label = 'High SAT '+str(self.HIGH_SAT_THRESHOLD))    
        # plt.hlines( self.HIGH_SAT_THRESHOLD, 0, len_x, color='k'   , alpha=0.5, linestyles='dashed', linewidth=1, 
        #             label = 'Low SAT '+str(self.HIGH_SAT_THRESHOLD))    
        # plt.hlines( self.PIXEL_RATIO_THRESHOLD, 0, len_x, color='red'  , alpha=0.5, linestyles='dashed', linewidth=1)    
        # plt.hlines( 200      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
        # plt.hlines( 150      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
        
        for (x,_,_) in self.SrcAdjustment_history:
            ax.plot(x,-20, 'bo') 
        leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
        leg.set_title('Legend',prop={'size':11})
        ax.set_title('Plot 4 - UNDIST - Pxl Thrshld: {:3d} OffCt Thrshld: {:3d}'.format( 
            self.PIXEL_RATIO_THRESHOLD, self.OFF_CENTER_ROI_THRESHOLD ))


    def plot_4W(self, legend = 'best', size=(15,7), pxlthr=False):
        plt.figure(figsize=size)
        ax = plt.gca()
        len_x = len(self.imgUndistStats['Hue'])

        ax.plot(self.imgWarpedStats['Hue'], color='r', label='Hue (W)')
        ax.plot(self.imgWarpedStats['Lvl'], color='g', label='Lvl (W)')
        ax.plot(self.imgWarpedStats['Sat'], color='b', label='Sat (W)')
        # ax.plot(self.imgWarpedStats['HLS'] ,color='k', label='HLS')
        ax.plot(self.imgWarpedStats['RGB'] ,color='darkorange', label='RGB (W)')
        ax.plot(self.imgUndistStats['RGB'] ,color='darkblue', alpha = 0.2, label='RGB (U)', linestyle='dashed')

        
        ax.plot(np.array(self.imgCondHistory), label = 'ImgCondition')
        # ax.plot(np.array(self.acceptHistory) , label = 'Polynom Acpt/Rjct')
        if pxlthr:
            ax.plot(self.imgPixelRatio, color='sienna', label = 'Pixel Rto')

        plt.hlines( self.HIGH_RGB_THRESHOLD    , 0, len_x, color='darkred', alpha=0.5, linestyles='dashed', linewidth=1, 
                    label = 'High RGB '+str(self.HIGH_RGB_THRESHOLD))    
        # plt.hlines( 175      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
        plt.hlines( self.MED_RGB_THRESHOLD , 0, len_x, color='darkgreen', alpha=0.5, linestyles='dashed', linewidth=1, 
                    label = 'Med RGB '+str(self.MED_RGB_THRESHOLD))    
        plt.hlines( self.LOW_RGB_THRESHOLD , 0, len_x, color='brown', alpha=0.5, linestyles='dashed', linewidth=1,  
                    label = 'Low RGB '+str(self.LOW_RGB_THRESHOLD))    
        # plt.hlines( 120      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
        plt.hlines( self.VLOW_RGB_THRESHOLD, 0, len_x, color='red'   , alpha=0.5, linestyles='dashed', linewidth=1,  
                    label = 'VLow RGB '+str(self.VLOW_RGB_THRESHOLD))    
        # plt.hlines( self.LOW_SAT_THRESHOLD , 0, len_x, color='g'   , alpha=0.5, linestyles='dashed', linewidth=1,
        #             label = 'High SAT '+str(self.HIGH_SAT_THRESHOLD))    
        # plt.hlines( self.HIGH_SAT_THRESHOLD, 0, len_x, color='k'   , alpha=0.5, linestyles='dashed', linewidth=1, 
        #             label = 'Low SAT '+str(self.HIGH_SAT_THRESHOLD))    
        # plt.hlines( self.PIXEL_RATIO_THRESHOLD, 0, len_x, color='red'  , alpha=0.5, linestyles='dashed', linewidth=1)    
        # plt.hlines( 200      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
        # plt.hlines( 150      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
        
        for (x,_,_) in self.SrcAdjustment_history:
            ax.plot(x,-10, 'bo') 
        leg = plt.legend(loc=legend,frameon=True, fontsize = 10, markerscale = 6)
        leg.set_title('Legend',prop={'size':11})
        ax.set_title('Plot 4W - WARPED - Pxl Thrshld: {:3d} OffCt Thrshld: {:3d}'.format(
             self.PIXEL_RATIO_THRESHOLD, self.OFF_CENTER_ROI_THRESHOLD ))



    def plot_5(self, legend = 'best', size=(15,7)):
        plt.figure(figsize=size)
        ax = plt.gca()
        len_x = len(self.imgUndistStats['RGB'])
        ax.plot(self.LeftLane.pixelRatio , color = 'r', label='Left Pxl Ratio') 
        ax.plot(self.RightLane.pixelRatio, color = 'b', label='Right Pxl Ratio') 
        ax.plot(self.imgPixelRatio       , color = 'k', label='Img Pxl Ratio') 
        ax.plot(self.imgWarpedStats['Sat'], label='Sat', linestyle='dashed'  )


        ax.plot(np.array(self.acceptHistory) , label = 'Polynom Acpt/Rjct')
        ax.plot(np.array(self.imgCondHistory), label = 'Normal/Dark')

        ax.set_title('Plot 5 - Pxl Thrshld: {:3d} OffCt Thrshld: {:3d}'.format( self.PIXEL_RATIO_THRESHOLD, self.OFF_CENTER_ROI_THRESHOLD ))
        # plt.hlines( self.RGB_MEAN_THRESHOLD , 0, len_x, color='blue' , alpha=0.5, linestyles='dashed', linewidth=1)    
        # plt.hlines( self.MED_RGB_THRESHOLD  , 0, len_x, color='darkgreen', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
        plt.hlines( self.LOW_RGB_THRESHOLD    , 0, len_x, color='blue' , alpha=0.5, linestyles='dashed', linewidth=1)    
        plt.hlines( self.VLOW_RGB_THRESHOLD   , 0, len_x, color='blue' , alpha=0.5, linestyles='dashed', linewidth=1)    
        plt.hlines( self.PIXEL_RATIO_THRESHOLD, 0, len_x, color='red'  , alpha=0.8, linestyles='dashed', linewidth=1, label='<30>')    
        # plt.hlines( self.OFF_CENTER_ROI_THRESHOLD, 0, len_x, color='black', alpha=0.8, linestyles='dashed', linewidth=1)    
        # plt.hlines(-self.OFF_CENTER_ROI_THRESHOLD, 0, len_x, color='black', alpha=0.8, linestyles='dashed', linewidth=1)    
        
        for (x,_,_) in self.SrcAdjustment_history:
            ax.plot(x, -10, 'bo') 
        leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
        leg.set_title('Legend',prop={'size':11})

    def plot_6(self, legend = 'best', size=(15,7), clip = 9999999):
        plt.figure(figsize=size)
        ax = plt.gca()
        len_x = len(self.imgUndistStats['RGB'])
        ax.plot(np.array( self.LeftLane.RSE_history), label = 'Left RSE ')
        ax.plot( np.clip( self.LeftLane.radius_history ,0, clip), color = 'r', label='Left Radius') 
        ax.plot( np.clip( self.RightLane.radius_history,0, clip), color = 'b', label='Right Radius') 
        # ax.plot(np.array(self.LeftLane.radius_history)  , color = 'r', label='Left Radius') 
        # ax.plot(np.array(self.RightLane.radius_history) , color = 'b', label='Right Radius') 
        
        plt.hlines( 10  , 0, len_x, color='red'  , alpha=0.8, linestyles='dashed', linewidth=1, label='<10>')    
        plt.hlines( self.PIXEL_RATIO_THRESHOLD   , 0, len_x, color='red'  , alpha=0.8, linestyles='dashed', linewidth=1, label='<30>')    

        plt.ylim(-25, 10000)
        for (x,_,_) in self.SrcAdjustment_history:
            ax.plot(x,-20, 'bo') 
        leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
        leg.set_title('Legend',prop={'size':11})
        ax.set_title('Plot 6 - Curvature - Pxl Thrshld: {:3d} OffCt Thrshld: {:3d}'.format(
             self.PIXEL_RATIO_THRESHOLD, self.OFF_CENTER_ROI_THRESHOLD ))





    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def set_thresholding_parms(self):
        '''
        select thresholding parameters based on current image condtiions
        currently we only compare the RGB mean value against a threshold
        other criteria can be considered
        '''
        # if (self.imgWarpedStats['Sat'][-1] >  self.HIGH_SAT_THRESHOLD) or \
        #    (self.imgWarpedStats['RGB'][-1] >  self.HIGH_RGB_THRESHOLD) :
        #     print('Over exposed frame - detection will be skipped')
        #     print(' WarpedStats[Sat]: {}  WarpedStats[RGB]: {} '.format(self.imgWarpedStats['Sat'][-1],
        #                                         self.imgWarpedStats['RGB'][-1]))
        #     self.skipFrameDetection = True
        #     self.slidingWindowBootstrap = True
        #     self.Conditions = 'overexposed'
        #     self.imgCondHistory.append(-40)
        #     thresholdParms = None

        # if (self.imgWarpedStats['Sat'][-1] >  self.HIGH_SAT_THRESHOLD) or \
        #    (self.imgWarpedStats['RGB'][-1] >  self.HIGH_RGB_THRESHOLD) :
        #     print('Over exposed frame  -  WarpedStats[Sat]: {:7.2f}  WarpedStats[RGB]: {:7.2f} '.format(
        #         self.imgWarpedStats['Sat'][-1], self.imgWarpedStats['RGB'][-1]))
        #     self.Conditions = 'hisat'
        #     self.imgCondHistory.append(30)
        # 
        # if (self.imgUndistStats['Sat'][-1] < self.LOW_SAT_THRESHOLD):
        #     self.Conditions = 'lowsat'
        #     self.imgCondHistory.append(-30)
        
        if (self.imgWarpedStats['RGB'][-1] < self.VLOW_RGB_THRESHOLD) :
            self.Conditions = 'vlow'
            self.imgCondHistory.append(-20)
        
        elif (self.imgWarpedStats['RGB'][-1] < self.LOW_RGB_THRESHOLD) :
            if (self.imgWarpedStats['Sat'][-1] < self.LOW_SAT_THRESHOLD):
                self.Conditions = 'lowsat'
                self.imgCondHistory.append(-30)
            else:
                self.Conditions = 'low'
                self.imgCondHistory.append(-10)
        
        elif (self.imgWarpedStats['RGB'][-1] < self.MED_RGB_THRESHOLD) :
            self.Conditions = 'med'
            self.imgCondHistory.append(0)

        elif  (self.imgWarpedStats['RGB'][-1] < self.HIGH_RGB_THRESHOLD) :
            self.Conditions = 'high'
            self.imgCondHistory.append(10)
        else:
            self.Conditions = 'xhigh'
            self.imgCondHistory.append(20)
        
        self.thresholdMethod  =  self.thresholdMethods[self.mode][self.Conditions]
        self.thresholdStrs    =  self.itStr[self.mode][self.Conditions]           
        self.thresholdParms   =  self.ImageThresholds[self.mode][self.Conditions]         
    
        return 

    def initialize_thresholding_parameters(self):

#       self.thresholdMethods[2]['high']   =  self.HIGH_THRESHOLDING   ## ('high_thresholding'    , 'cmb_mag_x')
#       self.thresholdMethods[2]['med']    =  self.NORMAL_THRESHOLDING ## ('normal_thresholding'  , 'cmb_rgb_lvl_sat')
#       self.thresholdMethods[2]['low']    =  self.LOW_THRESHOLDING    ## ('low_thresholding'     , 'cmb_mag_xy')
#       self.thresholdMethods[2]['vlow']   =  self.VLOW_THRESHOLDING   ## ('Vlow_thresholding'     , 'cmb_mag_xy')
#       self.thresholdMethods[2]['hisat']  =  self.HISAT_THRESHOLDING  ## ('hisat_thresholding'   , 'cmb_mag_x')
#       self.thresholdMethods[2]['lowsat'] =  self.LOWSAT_THRESHOLDING ## ('lowsat_thresholding'  , 'cmb_hue_x')

        ##---------------------------------------------
        ## Image Thresholding params 
        ##---------------------------------------------
        self.ImageThresholds  = { 1: {} , 2: {} }
        self.itStr            = { 1: {} , 2: {} }
        self.thresholdMethods = { 1: {} , 2: {} }

        self.thresholdMethods[1]['xhigh']   =  self.XHIGH_THRESHOLDING   
        self.thresholdMethods[1]['high']   =  self.HIGH_THRESHOLDING   
        self.thresholdMethods[1]['med']    =  self.NORMAL_THRESHOLDING 
        self.thresholdMethods[1]['low']    =  self.LOW_THRESHOLDING    
        self.thresholdMethods[1]['Vlow']   =  self.VLOW_THRESHOLDING    
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
        # self.thresholdMethods[2]['hisat']  =  self.HISAT_THRESHOLDING  
        # self.thresholdMethods[2]['lowsat'] =  self.LOWSAT_THRESHOLDING         

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

        for mode in [1,2]:
            for cond in self.ImageThresholds[mode].keys():
                self.itStr[mode][cond] = {}
                for thr in self.ImageThresholds[mode][cond].keys():
                    self.itStr[mode][cond][thr] = str(self.ImageThresholds[mode][cond][thr])




        # ax.plot(np.array(self.imgCondHistory), label = 'Normal/Dark')

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
        # ax.plot(np.array(self.acceptHistory), label = 'Polynom Acpt/Rjct')
        # ax.plot(np.array(self.imgCondHistory), label = 'Normal/Dark')

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
        # ax.plot(np.array(self.acceptHistory), label = 'Polynom Acpt/Rjct')
        # ax.plot(np.array(self.imgCondHistory), label = 'Normal/Dark')

        # result_1, _ = displayDetectedRegion(self.imgUndist, self.LeftLane.fitted_current, self.RightLane.fitted_current, 
        #                              self.Minv, start=region_zone_sep_pos        , beta = 0.2, color = polyRegionColor1)
        # result_1, _ = displayDetectedRegion(result_1      , self.LeftLane.fitted_current, self.RightLane.fitted_current, 
        #                              self.Minv, start=200,end=region_zone_sep_pos, beta = 0.2, color = polyRegionColor2)


