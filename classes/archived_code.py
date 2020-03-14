


"""
        ## Image Thresholding params ---------------
        # self.ksize                = 7
        # self.x_thr           = (30,255)
        # self.y_thr           = (70,255)
        # self.mag_thr              = (35,255)
        # self.dir_thr              = (40,65)
        # self.sat_thr              = (65,255)
        # self.lvl_thr              = (180, 255)
        # self.rgb_thr              = (180,255)
        ## Image Thresholding params (1/28/20) ------
        self.ksize      = 7
        self.x_thr = (30,255)
        self.y_thr = (70,255)
        self.mag_thr    = (35,255)
        self.dir_thr    = (40,65)
        self.sat_thr    = (110,255)
        self.lvl_thr    = (205, 255)
        self.rgb_thr    = (205,255)
        ##---------------------------------------------
       
       
        ##  Threshold params for Warped Image ----------
        # self.ksize_2              = 7
        # self.x_thr_2         = (30,255)
        # self.y_thr_2         = (70,255)
        # self.mag_thr_2            = (10,50) 
        # self.dir_thr_2            = (5,25)  
        # self.sat_thr_2            = (70, 255)
        # self.lvl_thr_2            = (180, 255)
        # self.rgb_thr_2            = (180,255)
        
        ## Warped Image Threshold params (1/28/20) ------
        self.ksize_2      = 7
        self.x_thr_2 = (30,255)
        self.y_thr_2 = (70,255)
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

"""




############################################################################################################
############################################################################################################
###       
###                                 Arhcived Methods for LINE class                                                                              
###      
############################################################################################################
############################################################################################################

"""
def fit_polynomial_V1(self, debug = False):
### Fit a second order polynomial to each using `np.polyfit` ### 
    try:
        self.current_fit = np.polyfit(self.ally , self.allx , 2, full=False)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('---------------------------------------------------')
        print('fitPolynomial(): The function failed to fit a line!')
        print(' current_fit will be set to best_fit               ')
        print('---------------------------------------------------')
        self.current_fit  = self.best_fit
        
    if debug:
        print('\nfit_polynomial:')
        print('-'*20)    
        print(' start fit history length        : ', len(self.fit_history), '  best_fit_history length   : ', len(self.best_fit_history))
        print(' previous fit                    : ', self.fit_history[-1])
        print(' current polyfit                 : ', self.current_fit)
        print(' current best_fit                : ', self.best_fit)
        print(' current avg_RSE :', self.best_avg_RSE, ' avg_diffs : ', self.fit_avg_RSE)
        
        
    if len(self.fit_history) > 0 :
        ### self.best_fit   = sum(self.fit_history)/ len(self.fit_history)  
        ### self.diffs      = self.current_fit - self.fit_history[-1]
        self.fit_diffs  = self.current_fit - self.fit_history[-1]     
        self.best_diffs = self.current_fit - self.best_fit     
        self.fit_RSE    = np.round(np.sqrt(np.sum(self.fit_diffs**2)),3)                      
        self.best_RSE   = np.round(np.sqrt(np.sum(self.best_diffs**2)),3)                      
        print(' fit-history not empty : ')
        print('     - RSE(current, best): ',self.best_diffs, ' - ',  self.best_RSE, 'best_avg_RSE: ', self.best_avg_RSE)
        print('     - RSE(current, prev): ',self.fit_diffs , ' - ',  self.fit_RSE , ' fit_avg_RSE: ', self.fit_avg_RSE)

        if self.fit_RSE > (self.fit_avg_RSE + 25):
            print('-------------------------------------------------------------------')
            print(' BIG difference between newly fitted polynomial and best fitted ...')
            print('-------------------------------------------------------------------')
            self.current_fit = np.copy(self.best_fit)
            self.best_diffs  = self.current_fit - self.best_fit     
            self.best_RSE    = np.round(np.sqrt(np.sum(self.best_diffs**2)),3)                        
            print('current fit set to best fit -- ')
            print('   best_diffs: ', self.best_diffs, ' best_RSE: ', self.best_RSE, 'avg_RSE    : ', self.best_avg_RSE)
            print('   fit_diffs : ', self.fit_diffs , ' fit_RSE : ', self.fit_RSE , 'fit_avg_RSE: ', self.fit_avg_RSE)
    
    else:
        self.fit_diffs  = self.current_fit  
        self.fit_RSE    = np.round(np.sqrt(np.sum(self.fit_diffs**2)),3)                        
        self.fit_avg_RSE= np.round(np.sqrt(np.sum(self.fit_diffs**2)),3)                        
        # self.best_fit = self.current_fit
        self.best_diffs = self.current_fit      
        self.best_RSE   = np.round(np.sqrt(np.sum(self.best_diffs**2)),3)                        
        self.best_avg_RSE    = np.round(np.sqrt(np.sum(self.best_diffs**2)),3)                        
        print(' fit history is empty -- ')
        print('     - RSE(current, best): ', self.best_RSE, 'best_avg_RSE: ', self.best_avg_RSE)
        print('     - RSE(current, prev): ', self.fit_RSE , ' fit_avg_RSE: ', self.fit_avg_RSE)
"""
"""
        # self.y_checkpoints = np.linspace(0,  self.height,  100, dtype = np.int)     
        # self.y_checkpoints = np.flip(np.sort(self.y_checkpoints))
        # self.y_checkpoints = np.concatenate((np.arange(self.y_src_bot,-1,-100), [self.y_src_top]))
        # self.y_checkpoints = np.flip(np.sort(self.y_checkpoints))
"""

"""
        # if len(self.curr_fit_history) > 0:
            # self.best_fit_diffs = self.curr_fit - self.best_fit_history[-1]
            # self.curr_fit_diffs = self.curr_fit - self.curr_fit_history[-1]
        # else:
            # self.best_fit_diffs = self.curr_fit - self.curr_fit      
            # self.curr_fit_diffs = self.curr_fit - self.curr_fit
            
        # self.proposed_best_RSE  = np.round(np.sqrt(np.sum(BestLaneDiffs**2)),3)                        
        # self.proposed_fit_RSE   = np.round(np.sqrt(np.sum(FitLaneDiffs**2)),3)                        

        # self.RSE   = np.round(np.sqrt(np.sum(self.best_fit_diffs**2)),3)                        

        # self.curr_RSE    = np.round(np.sqrt(np.sum(self.curr_fit_diffs**2)),3)                                



    # def get_slope_via_delta(self, y_eval = 0, xfitted = None, debug = False):
    #     '''
    #     Uses delta_x/delta_y to calculate slope
    #     Should yield same results as get_slope()
    #     '''
    #     if xfitted is None:
    #         xfitted = self.fitted_current[0,:]
    #     X1 = y_eval - 5
    #     X2 = y_eval + 5
    #     Y2 = xfitted[X2]
    #     Y1 = xfitted[X1]
    #     delta_x = X2 - X1
    #     delta_y = Y1 - Y2
    #     slope = delta_x / delta_y
    #     # slope  = (2*A*(y_eval))+B
    #     # slope_pi = slope + np.pi/2
    #     slope1  = np.round(np.rad2deg(np.arctan(slope)),3)

    #     if debug:
    #         print(' get_slope(): Dx: ',delta_x, 'Dy: ',delta_y ,' slope', slope, ' slope1: ', slope1, ' (deg) - based on pixels')
    #     return slope1

    # def get_slope_mtrs(self, y_eval = 0, fit_parms = None, debug = False):
    #     if fit_parms is None:
    #         fit_parms = self.curr_fit
    #     A,B,_ = fit_parms 
    #     A1 = (A * self.MX)/ (self.MY**2)
    #     B1 = (B * self.MX / self.MY)

    #     slope = (2*A1*(y_eval * self.MY))+B1
    #     slope_pi = slope + np.pi/2
    #     slope1 = np.round(np.rad2deg(np.arctan(slope_pi)),3)

    #     if debug:
    #         print(' get_slope(): x: {}  y:{}  slope', slope, ' slope_pi: ', slope_pi, ' slope1: ', slope1, ' (deg) - based on pixels')
    #     return slope1


    # def get_radius_v1(self, y_eval = 0 , fit_parms = None, debug = False):
    #     if fit_parms is None:
    #         fit_parms = self.curr_fit

    #     # assert units in ['m', 'p'], "Invalid units parameter, must be 'm' for meters or 'p' for pixels"
    #     # def radius(y_eval, fit_coeffs, units, debug = False):
    #     # MY = 30/720   (self.MY_denom)  # meters per pixel in y dimension
    #     # MX= 3.7/700   (self.MX_denom)  # meters per pixel in x dimension
    #     A,B,_ = fit_parms 

    #     A = (A * self.MX)/ (self.MY**2)
    #     B = (B * self.MX / self.MY)
    #     radius = ((1 + ((2*A*(y_eval * self.MY))+B)**2)** 1.5)/np.absolute(2*A) 
    #     if debug:
    #         print(' get_radius(): ', radius, ' (m)')

    #     return np.round(radius,4) 
"""


############################################################################################################
############################################################################################################
###       
###                                 Archived methods for PIPELINE class                                                                              
###      
############################################################################################################
############################################################################################################


"""
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
                    
"""

"""
        # self.x_src_top_left           = kwargs.get('x_src_top_left'          , self.x_src_top_left )  ## 580 -> 573
        # self.x_src_top_right          = kwargs.get('x_src_top_right'         , self.x_src_top_right)
        # self.x_src_bot_left           = kwargs.get('x_src_bot_left'          , self.x_src_bot_left ) 
        # self.x_src_bot_right          = kwargs.get('x_src_bot_right'         , self.x_src_bot_right) 
        # self.y_src_top                = kwargs.get('y_src_top'               , self.y_src_top      )  ## 460 -> 465 y_src_bot - 255
        # self.y_src_bot                = kwargs.get('y_src_bot'               , self.y_src_bot      )  ## image.shape[0] - 20
        # self.x_dst_left               = kwargs.get('x_dst_left'              , self.x_dst_left     )
        # self.x_dst_right              = kwargs.get('x_dst_right'             , self.x_dst_right    )
        # self.y_dst_top                = kwargs.get('y_dst_top'               , self.y_dst_top      )
        # self.y_dst_bot                = kwargs.get('y_dst_bot'               , self.height - 1     )   
        # self.src_points_list          = [ (self.x_src_top_left , self.y_src_top),
        #                                   (self.x_src_top_right, self.y_src_top), 
        #                                   (self.x_src_bot_right, self.y_src_bot),
        #                                   (self.x_src_bot_left , self.y_src_bot)]
        # self.src_points               = np.array(self.src_points_list, dtype = np.float32)
        # self.dst_points_list          = [ (self.x_dst_left , self.y_dst_top), 
        #                                   (self.x_dst_right, self.y_dst_top), 
        #                                   (self.x_dst_right, self.y_dst_bot), 
        #                                   (self.x_dst_left , self.y_dst_bot)]
        # self.dst_points               = np.array(self.dst_points_list, dtype = np.float32)
"""

"""
        ###----------------------------------------------------------------------------------------------
        ### Apply Thresholding based on self.mode parameter
        ###----------------------------------------------------------------------------------------------
        # if self.mode == 1:

            # if (image_to_Threshold.man() > self.RGB_MEAN_THRESHOLD) :
            #     self.NormalConditions = True
            #     self.thresholdResult =  self.NORMAL_THRESHOLDING
            #     thresholdParms  =  self.ImageNormalThresholds 
            # else:
            #     self.NormalConditions = False
            #     self.thresholdResult = self.DARK_THRESHOLDING
            #     thresholdParms  =  self.ImageDarkThresholds
            # output = apply_thresholds(self.imgUndist, thresholdParms, debug = debug)
            # self.imgThrshld = output[self.thresholdResult]
            # self.imgThrshldWarped, _, Minv = perspectiveTransform(self.imgThrshld, self.src_points, self.dst_points, debug = )
            # working_image = self.imgThrshldWarped; sfx = '_thr_wrp'   ### Warped AFTER thresholding
            
        # else:
            # if (self.WarpedRGBMean[-1] > self.RGB_MEAN_THRESHOLD) :
            #     self.NormalConditions = True
            #     self.thresholdResult =  self.NORMAL_THRESHOLDING
            #     thresholdParms  =  self.WarpedNormalThresholds 
            # else:
            #     self.NormalConditions = False
            #     self.thresholdResult = self.DARK_THRESHOLDING
            #     thresholdParms  =  self.WarpedDarkThresholds
            # output  = apply_thresholds(self.imgWarped, thresholdParms, debug = debug2)
            # self.imgWarpedThrshld = output[self.thresholdResult]                   
            # working_image = self.imgWarpedThrshld; sfx = '_wrp_thr'   ### Warped BEFORE thresholding
"""
"""
        #----------------------------------------------------------------------------
        ## Source/Dest points for Perspective Transform   
        #----------------------------------------------------------------------------
        # self.x_src_top_left       = kwargs.get('x_src_top_left' ,  575)
        # self.x_src_top_right      = kwargs.get('x_src_top_right',  708)
        # self.x_src_bot_left       = kwargs.get('x_src_bot_left' ,  220) 
        # self.x_src_bot_right      = kwargs.get('x_src_bot_right', 1090) 
        # self.y_src_top            = kwargs.get('y_src_top'      ,  460)  ## y_src_bot - 255
        # self.y_src_bot            = kwargs.get('y_src_bot'      ,  700)  ## image.shape[0] - 20
        
        ## 02-05-2020----------------------------------
        # self.x_src_top_left       = kwargs.get('x_src_top_left' ,  630)
        # self.x_src_top_right      = kwargs.get('x_src_top_right',  720)
        # self.x_src_bot_left       = kwargs.get('x_src_bot_left' ,  295)
        # self.x_src_bot_right      = kwargs.get('x_src_bot_right', 1105)
        # self.y_src_top            = kwargs.get('y_src_top'      ,  470)
        # self.y_src_bot            = kwargs.get('y_src_bot'      ,  700)  
        #----------------------------------------------
        # self.x_src_top_left       = kwargs.get('x_src_top_left' ,   600)  ## 580 -> 573
        # self.x_src_top_right      = kwargs.get('x_src_top_right',   740)
        # self.x_src_bot_left       = kwargs.get('x_src_bot_left' ,   295) 
        # self.x_src_bot_right      = kwargs.get('x_src_bot_right',  1105) 
        #----------------------------------------------
"""
"""
 adjustRoIWindow()
        ###----------------------------------------------------------------------------------------------
        # if minimum radius of lanes is less than 250 meters, adjust perspective transformation points  
        ###----------------------------------------------------------------------------------------------       
        # if min_radius < 250:
        #     # self.y_src_top = self.y_src_top + 10
        #     print('-'*35)
        #     print(' Adjust warping source points -  min_raidus: {}  OffCtr Pxls: {}    max_diffs: {}    imgPxlRatio: {} '.format(min_radius, 
        #             off_center_pixels, max_diffs, self.imgPixelRatio))
        #     print('-'*35)
        #     print('   Current  src_points_list : ', self.prev_src_points_list)
        #     print('        mod_src_points_list : ', self.mod_src_points_list)
        #     print('        dyn_src_points_list : ', self.dyn_src_points_list)
        # #   print()
        # #   print('       New  src_points_list : ', self.src_points_list)
        # #   print('           Prev Left x_base : ', self.LeftLane.x_base[-2], '   Right x_base  :', self.RightLane.x_base[-2])
        # #   print('           New  Left x_base : ', self.LeftLane.x_base[-1], '   Right x_base  :', self.RightLane.x_base[-1])
        # #   print()


    def debugInfo_ImageInfo(self):
        # print(' F: {:5.0f}- Undist- RGBTh: {:6.2f}  RGB: {:6.2f}  S: {:6.2f}  H: {:6.2f}  L: {:6.2f}'\
        #         '  C: {:6s}-{:10s}  Warp-  RGB: {:6.2f}  S: {:6.2f}  H: {:6.2f}  L: {:6.2f}'\
        #         '  C: {:6s}-{:10s}'.format( self.inVideo.currFrameNum, self.low_rgb_threshold, 
        #         self.imgUndistStats['RGB'][-1], self.imgUndistStats['Sat'][-1], self.imgUndistStats['Hue'][-1], 
        #         self.imgUndistStats['Lvl'][-1], self.Conditions , self.thresholdResult, self.imgWarpedStats['RGB'][-1], 
        #         self.imgWarpedStats['Sat'][-1], self.imgWarpedStats['Hue'][-1], self.imgWarpedStats['Lvl'][-1], 
        #         self.Conditions ,  self.thresholdResult))

"""
 


"""
    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def assess_lane_detection_results(self):

        # if  self.slidingWindowBootstrap or self.RoIAdjustment:
            # self.fitPolynomials = True
            # return 

        self.fitPolynomials = self.findLanePixelsRC

        for Lane in [self.LeftLane, self.RightLane]:

        if (self.LeftLane.pixelCount < self.PIXEL_THRESHOLD) or (self.RightLane.pixelCount < self.PIXEL_THRESHOLD):
            print()
            print('-'*100)
            print(' Lane pixels count under threshold ')
            print(' leftPixelCount: {:7.2f}  or  rightPixelCount: {:7.2f}  <  Count Min Threshold: ({:4d}) '.format(
                      self.LeftLane.pixelCount, self.RightLane.pixelCount, self.PIXEL_THRESHOLD))
            print('-'*100)
            self.fitPolynomials = False

        elif (self.LeftLane.pixelRatio[-1]  < self.LANE_RATIO_THRESHOLD) or \
             (self.RightLane.pixelRatio[-1] < self.LANE_RATIO_THRESHOLD) : 
            print()
            print('-'*100)
            print(' Lane pixel Ratio under Threshold ')
            print(' left Pxl Ratio: {:7.2f}  or  right Pxl Ratio: {:7.2f}  < Lane Threshold: ({:4d})  OR  imgPixelRatio: {:7.2f} < ImagePxl Thr:({:4d})'.
                     format( self.LeftLane.pixelRatio[-1], self.RightLane.pixelRatio[-1], self.LANE_RATIO_THRESHOLD, 
                            self.imgPixelRatio[-1], self.PIXEL_RATIO_THRESHOLD))
            print(' leftPixelCount: {:7.2f}  or  rightPixelCount: {:7.2f}  <  Count Min Threshold: ({:4d}) '.format(
                      self.LeftLane.pixelCount, self.RightLane.pixelCount, self.PIXEL_THRESHOLD))
            print('-'*100)
            self.fitPolynomials = False

        # if (self.imgPixelRatio[-1]< self.PIXEL_RATIO_THRESHOLD) and (self.imgWarpedStats['Sat'][-1] > self.HIGH_SAT_THRESHOLD):
        elif (self.imgWarpedStats['RGB'][-1]> self.HIGH_RGB_THRESHOLD) and (self.imgWarpedStats['Sat'][-1] > self.HIGH_SAT_THRESHOLD):
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
            self.fitPolynomials = False
        



    ##--------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------
    def fit_polynomial_process_v1(self):
        ## If lane pixel detection was successful, try to fit polynomials over detected pixels 
         
        if  self.fitPolynomials:
            
            self.LeftLane.fit_polynomial(debug  = self.debug)
            self.RightLane.fit_polynomial(debug = self.debug)

            ### assess goodness of fitted polynomials
            poly_1 = self.LeftLane.assessFittedPolynomial(debug = self.debug)
            poly_2 = self.RightLane.assessFittedPolynomial(debug = self.debug)

            ## Realignment of the perspective transformation window will reuslt in a 
            ## High RSE Error. We will allow this error rate when it is a result of a 
            ## RoI realignment. Other wise proceed nornally.
            if  (self.slidingWindowBootstrap and self.RoIAdjustment):
                self.acceptPolynomial = True
                self.slidingWindowBootstrap  = False
                self.LeftLane.reset_best_fit(debug = self.debug)
                self.RightLane.reset_best_fit(debug = self.debug)
                msg = '\n RoIAdjustment performed - Polynomial fit will be accepted \n'
            else:
                self.acceptPolynomial = poly_1 and poly_2
                msg = '\n acceptPolynomial = poly_left ({}) and poly_right({}) = {}'.format(
                    poly_1, poly_2, poly_1 and poly_2)
        
        else:
            msg = '\n lane detection failed on frame: {}'.format(self.frameTitle)
            self.acceptPolynomial = False
        
        if self.debug :
            print(msg)        

        ### Process accepted or rejected polynomials 

        if self.acceptPolynomial:
            self.LeftLane.acceptFittedPolynomial(debug = self.debug, debug2 = self.debug2)
            self.RightLane.acceptFittedPolynomial(debug = self.debug, debug2 = self.debug2)
            self.ttlAcceptedFrames += 1
            self.ttlRejectedFramesSinceAccepted = 0
            self.ttlAcceptedFramesSinceRejected += 1
            self.validDetections   = True
            self.polyRegionColor1 = 'green'
            self.acceptHistory.append(0)
            # if self.slidingWindowBootstrap:
            self.slidingWindowBootstrap  = False
        else:
            self.ttlRejectedFrames += 1
            self.ttlAcceptedFramesSinceRejected = 0 
            self.ttlRejectedFramesSinceAccepted += 1
            self.LeftLane.rejectFittedPolynomial(debug = self.debug2)
            self.RightLane.rejectFittedPolynomial(debug = self.debug2)            
            self.slidingWindowBootstrap  = True       
            self.validDetections  = True
            self.polyRegionColor1 = 'yellow' 
            self.acceptHistory.append(-10)

            if self.ttlRejectedFramesSinceAccepted > self.YELLOW_DETECTION_LIMIT:
                self.slidingWindowBootstrap  = True        
                self.polyRegionColor1 = 'salmon'       
                self.acceptHistory.append(-20)
            
            if self.ttlRejectedFramesSinceAccepted > self.RED_DETECTION_LIMIT:
                self.slidingWindowBootstrap  = True
                self.validDetections  = False
                self.polyRegionColor1 = 'red'       
                self.acceptHistory.append(-30)

        ### Display debug info
        if self.debug: 

            if self.acceptPolynomial:
                print('\n => ACCEPT propsed polynomials  for {:5d}  Accepted frames Since Last Rejected - Left: {:3d}  Right: {:3d} \n'.format(
                    self.inVideo.currFrameNum, self.LeftLane.ttlAcceptedFramesSinceRejected, self.RightLane.ttlAcceptedFramesSinceRejected))
            else:

                print('\n => REJECT proposed polynomials for {:5d}  Rejected frames Since Last Detected - Left: {:3d}  Right: {:3d} \n'.format(  
                    self.inVideo.currFrameNum, self.LeftLane.ttlRejectedFramesSinceDetected, self.RightLane.ttlRejectedFramesSinceDetected))

            self.debugInfo_DetectionInfo()






"""


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
#                                          x_thr = x_thr, y_thr = y_thr, 
#                                        mag_thr = (50,255), dir_thr = (0,10), 
#                                        sat_thr = (80, 255), debug = False)
#     imgWarpedThrshld = imgWarpedThrshldList[-1]
#     print(imgWarpedThrshld.shape)
#     display_two( imgWarpedThrshld, imgThrshldWarped, 
#                  title1 = 'Warped BEFORE Thresholding', 
#                  title2 = 'Image warped AFTER Thresholding', winttl = frameTitle)
################################################################################################
