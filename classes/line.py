import os,sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import pprint 

pp = pprint.PrettyPrinter(indent=2, width=100)


# pp.pprint(sys.modules)
print(os.getcwd())
# pp.pprint(sys.path)

# Define a class to receive the characteristics of each line detection
class Line():

    def __init__(self, history, height, y_src_top, y_src_bot):

        self.history = history
        # height of current image 
        self.set_height(height)

        self.units = 'm'
        __MX_nom   = 3.7
        __MY_nom   =  30
        __MX_denom = 700
        
        self.set_MY(__MY_nom, self.height) # meters per pixel in y dimension
        self.set_MX(__MX_nom, __MX_denom) # meters per pixel in x dimension
        
        # was the line detected in the last iteration?
        self.detected = False  
        self.framesSinceDetected = 0 
        self.y_src_bot = y_src_bot
        self.y_src_top = y_src_top
        
        # y values correspoinding to current fitting/ image height
        self.set_ploty()
        self.y_checkpoints = np.concatenate((np.arange(y_src_bot,-1,-100), [y_src_top]))
        self.y_checkpoints = np.flip(np.sort(self.y_checkpoints))
        print(' y_checkpoints: ', self.y_checkpoints)

        #polynomial coefficients for the most recent fit / fit history
        self.current_fit = None ## np.array([0,0,0], dtype='float')        
        self.fit_history = deque([], self.history)

        #difference in fit coefficients between last and new fits
        self.fit_diffs = None ## np.array([0,0,0], dtype='float') 
        self.fit_diffs_history = deque([], self.history)
        
        self.fit_RSE            = 0 
        self.fit_avg_RSE        = 0
        self.fit_prev_avg_RSE   = 0
        self.fit_RSE_history    = deque([], self.history)

        # x values of the most recent fitting of the line
        self.current_xfitted = None        
        # x values of the last n fits of the line (including most current)
        self.xfitted_history = deque([], self.history)
        # Average of all x values for last n fits -- testing showed this is veery similar
        # to the result of best_xfitted.
        self.xfitted_avg     = None  

        #-----------------------------------------------------------------------
        # Best fit polynomial coefficients, diff with current fitted and history
        #-----------------------------------------------------------------------
        self.best_fit   = np.array([0,0,0], dtype='float') 
        self.best_fit_history   = deque([], self.history)
        
        self.best_diffs = None
        self.best_diffs_history = deque([], self.history)
        
        self.best_RSE           = 0  
        self.best_avg_RSE       = 0
        self.best_prev_avg_RSE  = 0
        self.best_RSE_history   = deque([], self.history)

        #average x values of the fitted line over the last n iterations
        self.best_xfitted          = None  
        self.best_xfitted_history  = deque([], self.history)
        
        #radius of curvature of the line in some units
        self.radius = deque([], self.history)
        
        #distance in meters of vehicle center from the line
        self.line_base_pixels = deque([], self.history)
        self.line_base_meters = deque([], self.history)

        #radius of curvature of the line in some units
        self.slope = deque([], self.history)
        
        #x values for detected line pixels
        self.allx = None  

        #y values for detected line pixels
        self.ally = None  
        
        print(' Line init() complete ')

    def set_MX(self, nom = 3.7, denom = 700, debug = False):
        self.MX_nom   = nom
        self.MX_denom = denom
        self.MX = self.MX_nom/self.MX_denom
        if debug:
            print(' MX                : {} '.format(self.MX))
            print(' MX_nominator   is : {} '.format(self.MX_nom))
            print(' MX_denominator is : {} '.format(self.MX_denom))
        return self.MX 

        
    def set_MY(self, nom = 30, denom = 720, debug = False):
        self.MY_nom   = nom
        self.MY_denom = denom
        self.MY = self.MY_nom/self.MY_denom
        if debug:
            print(' MY                : {} '.format(self.MY))
            print(' MY_nominator   is : {} '.format(self.MY_nom))
            print(' MY_denominator is : {} '.format(self.MY_denom))
        return self.MY
    
    def set_height(self, height):
        self.height = height

    def set_ploty(self, start = 0, end = 0):
        if end == 0:
            end = self.height
        self.ploty = np.linspace(start, end-1, end-start,dtype = np.int)
                
    def set_linePixels(self, x_values, y_values):
        self.allx = x_values
        self.ally = y_values
        
    def set_line_base_pos(self, y_eval = 0, xfitted = None, debug = False):
        lb_pxls, lb_mtrs = self.get_line_base_pos(y_eval, xfitted, debug = debug)
        self.line_base_pixels.append( lb_pxls )
        self.line_base_meters.append( lb_mtrs )

    def get_line_base_pos(self, y_eval= 0 , xfitted = None, debug = False):
        if xfitted is None:
            xfitted = self.current_xfitted
        
        # self.line_base_pos = round(self.xfitted_history[-1][y_eval] * self.MX,3)
        lb_pixels =  round(xfitted[y_eval],3) 
        lb_meters =  round(xfitted[y_eval] * self.MX,3)

        # A,B, C = self.best_fit   
        # A = (A * self.MX)/ (self.MY**2)
        # B = (B * self.MX / self.MY)
        # lb_meters = A * ((y_eval * self.MY)**2) + B * (y_eval * self.MY) + C
        if debug: 
            print(' get_line_base_pos():  y_eval: {} current_xfitted[y_eval]: {}    MX: {}    lb_meters: {} '.format(y_eval, 
                    xfitted[y_eval] , self.MX, lb_meters))
        return lb_pixels, lb_meters
               

    def set_slope(self, y_eval = 0, fit_parms = None, debug = False):
        slope = self.get_slope(y_eval, fit_parms, debug = debug)
        self.slope.append(slope)

    def get_slope(self, y_eval = 0, fit_parms = None, debug = False):
        if fit_parms is None:
            fit_parms = self.current_fit
        A,B,_ = fit_parms 

        slope  = ((2*A*(y_eval))+B)
        
        slope1  = np.round(np.rad2deg(np.arctan(slope)),3)
        slope2  = slope1 + 90.0
        if debug:
            print(' get_slope(): x: {}  y:{}  slope', slope, ' slope1: ', slope1, ' slope2: ', slope2, ' (deg) - based on pixels')
        return slope2
        

    def get_slope_via_delta(self, y_eval = 0, xfitted = None, debug = False):
        '''
        Uses delta_x/delta_y to calculate slope
        Should yield same results as get_slope()
        '''
        if xfitted is None:
            xfitted = self.current_xfitted
        X1 = y_eval - 5
        X2 = y_eval + 5
        Y2 = xfitted[X2]
        Y1 = xfitted[X1]
        delta_x = X2 - X1
        delta_y = Y1 - Y2
        slope = delta_x / delta_y
        # slope  = (2*A*(y_eval))+B
        # slope_pi = slope + np.pi/2
        slope1  = np.round(np.rad2deg(np.arctan(slope)),3)

        if debug:
            print(' get_slope(): Dx: ',delta_x, 'Dy: ',delta_y ,' slope', slope, ' slope1: ', slope1, ' (deg) - based on pixels')
        return slope1



    def get_slope_mtrs(self, y_eval = 0, fit_parms = None, debug = False):
        if fit_parms is None:
            fit_parms = self.current_fit
        A,B,_ = fit_parms 
        A1 = (A * self.MX)/ (self.MY**2)
        B1 = (B * self.MX / self.MY)

        slope = (2*A1*(y_eval * self.MY))+B1
        slope_pi = slope + np.pi/2
        slope1 = np.round(np.rad2deg(np.arctan(slope_pi)),3)

        if debug:
            print(' get_slope(): x: {}  y:{}  slope', slope, ' slope_pi: ', slope_pi, ' slope1: ', slope1, ' (deg) - based on pixels')
        return slope1

    def set_curvature(self, y_eval = 0 , fit_parms = None, debug = False):
        curvature = self.get_curvature(y_eval, fit_parms, debug = debug)
        self.radius.append( curvature )

    def get_curvature(self, y_eval = 0 , fit_parms = None, debug = False):
        if fit_parms is None:
            fit_parms = self.current_fit

        # assert units in ['m', 'p'], "Invalid units parameter, must be 'm' for meters or 'p' for pixels"
        # def radius(y_eval, fit_coeffs, units, debug = False):
        # MY = 30/self.MY_denom  # meters per pixel in y dimension
        # MX= 3.7/self.MX_denom  # meters per pixel in x dimension
        A,B,_ = fit_parms 
        
        A = (A * self.MX)/ (self.MY**2)
        B = (B * self.MX / self.MY)
        curvature = ((1 + ((2*A*(y_eval * self.MY))+B)**2)** 1.5)/np.absolute(2*A) 
        if debug:
            print(' get_curvature(): ', curvature, ' (m)')
        return np.round(curvature,2) 



      
    def fit_polynomial(self, debug = False):
        try:
            self.current_fit = np.polyfit(self.ally , self.allx , 2, full=False)
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('---------------------------------------------------')
            print('fitPolynomial(): The function failed to fit a line!')
            print(' current_fit will be set to best_fit               ')
            print('---------------------------------------------------')

        self.current_xfitted  = (self.current_fit[0]* self.ploty**2) + (self.current_fit[1]*self.ploty) + self.current_fit[2]
        
        self.current_slope     = self.get_slope(self.y_checkpoints,self.current_fit)
        self.current_curvature = self.get_curvature(self.y_checkpoints, self.current_fit)
        self.current_linepos   = self.current_xfitted[self.y_checkpoints]
        
        if debug:
            print('\nfit_polynomial:')
            print('-'*20)
            print(' New fitted polynomial  : ', self.current_fit)
            print(' self.current_xfitted   : ', self.current_xfitted.shape)
            # print(' self.current_slope     : ', self.current_slope.shape)
            # print(' self.current_slope_mtrs: ', self.current_slope_mtrs.shape)
            # print(' self.current_curvature : ', self.current_curvature.shape)
            # print(' self.current_linepos   : ', self.current_linepos.shape)
            print() 
        
    def acceptFittedPolynomial(self, debug= False):
    
        if debug:
            print('\nacceptFittedLine():')
            print('-'*20)    
            print(' start fit hist length  : ', len(self.fit_history), '  best_fit_history length   : ', len(self.best_fit_history))
            if len(self.fit_history) > 0 :   
                print(' previous fit           : ', self.fit_history[-1])
            print(' current polyfit        : ', self.current_fit)
            print(' current best_fit       : ', self.best_fit)
            print(' Fit  RSE(current, prev): {:10.3f}   prev avg_RSE: {:10.3f}    fit_avg_RSE:  {:10.3f} '.format( self.fit_RSE ,
                    self.fit_prev_avg_RSE , self.fit_avg_RSE))
            print(' Best RSE(current, best): {:10.3f}   prev avg_RSE: {:10.3f}    best_avg_RSE: {:10.3f} '.format( self.best_RSE, 
                    self.best_prev_avg_RSE, self.best_avg_RSE))
        
        self.detected = True
        self.framesSinceDetected = 0 
        
        self.set_fit(debug = debug)
        self.set_best_fit(debug = debug)
        self.set_xfitted()
        self.set_best_xfitted()
        
        self.set_lane_stats()
        
        if debug: 
            print()
            print(' end fit history length : ', len(self.fit_history), ' end best_fit_history length   : ', len(self.best_fit_history))
            print(' Accepted Polynomial Fit: ', self.fit_history[-1])
            print(' New Best fit           : ', self.best_fit)
            print(' Best fit diff          : ', self.best_diffs)
            print(' Fit  RSE(current, prev): {:10.3f}   prev avg_RSE: {:10.3f}    fit_avg_RSE:  {:10.3f} '.format( self.fit_RSE ,
                    self.fit_prev_avg_RSE , self.fit_avg_RSE))
            print(' Best RSE(current, best): {:10.3f}   prev avg_RSE: {:10.3f}    best_avg_RSE: {:10.3f} '.format( self.best_RSE, 
                    self.best_prev_avg_RSE, self.best_avg_RSE))

    def rejectFittedPolynomial(self, debug= False):
    
        self.detected = False
        self.framesSinceDetected += 1         
        self.set_lane_stats()

        if debug:
            print('\nrejectFittedLine():')
            print('-'*20)    
            print()
            print(' frames since last detected : ', self.framesSinceDetected)
            # print(' start fit hist length  : ', len(self.fit_history), '  best_fit_history length   : ', len(self.best_fit_history))
            # if len(self.fit_history) > 0 :   
                # print(' previous fit           : ', self.fit_history[-1])
            # print(' current polyfit        : ', self.current_fit)
            # print(' current best_fit       : ', self.best_fit)
            # print(' Fit  RSE(current, prev): {:10.3f}   prev avg_RSE: {:10.3f}    fit_avg_RSE:  {:10.3f} '.format( self.fit_RSE ,
                    # self.fit_prev_avg_RSE , self.fit_avg_RSE))
            # print(' Best RSE(current, best): {:10.3f}   prev avg_RSE: {:10.3f}    best_avg_RSE: {:10.3f} '.format( self.best_RSE, 
                    # self.best_prev_avg_RSE, self.best_avg_RSE))
        
        

            
    def set_lane_stats(self, debug = False):
        self.set_slope(self.y_src_bot, self.best_fit, debug = debug)
        self.set_curvature(self.y_src_bot, self.best_fit, debug = debug)
        self.set_line_base_pos( self.y_src_bot, self.best_xfitted, debug = debug)

        self.best_slope       = self.get_slope(self.y_checkpoints,self.best_fit)
        self.best_curvature   = self.get_curvature(self.y_checkpoints, self.best_fit)
        self.best_linepos     = self.best_xfitted[self.y_checkpoints]
            
    def set_fit(self, debug = False):
    
        self.fit_history.append(self.current_fit)
        
        if len(self.fit_history) > 1:
            self.fit_diffs = self.fit_history[-1] - self.fit_history[-2]
        else:
            self.fit_diffs = self.current_fit
        
        self.fit_diffs_history.append(self.fit_diffs)

        self.fit_RSE    = np.round(np.sqrt(np.sum(self.fit_diffs**2)),3)                                
        self.fit_RSE_history.append(self.fit_RSE)
        
        self.fit_prev_avg_RSE = self.fit_avg_RSE
        self.fit_avg_RSE = np.round(sum(self.fit_RSE_history)/ len(self.fit_RSE_history),3)
        
        if debug: 
            print(' set_fit()')
            if len(self.fit_history) > 1 :
                print('   Previous fit       : ', self.fit_history[-2])
            print('   New/Accepted       : ', self.fit_history[-1])
            print('   fit Diff           : ', self.fit_diffs)
            print('   fit history length : ', len(self.fit_history), ' fit_diffs_history length   : ', len(self.fit_diffs_history))
            print('   fit RSE(current,prev) : ', self.fit_RSE, '   fit_avg_RSE: ', self.fit_prev_avg_RSE, ' new fit_avg RSE:', self.fit_avg_RSE) 
 
    def set_best_fit(self, debug = False):
        ## Calculate new best_fit, using recently added polynomial.
        self.best_fit   = sum(self.fit_history)/ len(self.fit_history) 
        self.best_fit_history.append(self.best_fit)

        if len(self.best_fit_history) > 1:
            self.best_diffs = self.best_fit_history[-1] - self.best_fit_history[-2]
        else:        
            self.best_diffs = self.best_fit        
        
        self.best_diffs_history.append(self.best_diffs)
        
        self.best_RSE   = np.round(np.sqrt(np.sum(self.best_diffs**2)),3)                        
        self.best_RSE_history.append(self.best_RSE)
        
        self.best_prev_avg_RSE = self.best_avg_RSE
        self.best_avg_RSE      = np.round(sum(self.best_RSE_history)/ len(self.best_RSE_history),3)
                
        if debug: 
            print(' set_best_fit()')
            if len(self.best_fit_history) > 1 :
                print('   Previous best fit      : ', self.best_fit_history[-2])
            print('   New best fit           : ', self.best_fit_history[-1], '  ' , self.best_fit)
            print('   diff(previous, new)    : ', self.best_diffs)
            print('   best_fit_history       : ', len(self.best_fit_history), ' best_diffs_history: ', len(self.best_diffs_history))
            print('   best RSE(current,best) : ', self.best_RSE, '   best avg_RSE: ', self.best_prev_avg_RSE, ' new best avg RSE:', self.best_avg_RSE) 
                

    def set_xfitted(self, debug = False):
        self.current_xfitted  = (self.current_fit[0]* self.ploty**2) + (self.current_fit[1]*self.ploty) + self.current_fit[2]
        self.xfitted_history.append(self.current_xfitted)        
        self.xfitted_avg = np.sum(self.xfitted_history, axis = 0)/ len(self.xfitted_history)
        if debug: 
            print(' set_xfitted() - length: ', len(self.xfitted_history), ' shape: ', self.current_xfitted.shape)
            
    def set_best_xfitted(self, debug = False):
        self.best_xfitted      = (self.best_fit[0]*self.ploty**2) + (self.best_fit[1]*self.ploty) +self.best_fit[2]
        self.best_xfitted_history.append(self.best_xfitted)        
        if debug: 
            print(' set_best_xfitted() - length: ', len(self.best_xfitted_history), ' shape: ', self.best_xfitted.shape)
"""
    def set_slope_old(self, y_eval = 0, xfitted = None, debug = False):
        slope = self.get_slope(y_eval, xfitted, debug = debug)
        self.slope.append(slope)

    def get_slope_old(self, y_eval = 0, xfitted = None, debug = False):
        if xfitted is None:
            xfitted = self.current_xfitted
        slope = np.round(np.rad2deg(np.arctan2(y_eval,xfitted[y_eval])),3)
        if debug:
            print(' get_slope(): x: {}  y:{}  y/x', slope, ' (deg)')
        return slope
"""

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

