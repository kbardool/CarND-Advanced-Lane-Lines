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
    
    def __init__(self, history, height, y_src_top, y_src_bot, **kwargs):

        self.history       = history
        self.name          = kwargs.get('name', '')
        self.POLY_DEGREE   = kwargs.get('poly_degree', 2)
        self.RSE_THRESHOLD = kwargs.get('rse_threshold', 120)
        # height of current image 
        self.set_height(height)
        print(' poly degree = ', self.POLY_DEGREE)
        self.units       = 'm'
        
        __MX_nom         = 3.7
        __MY_nom         =  30
        __MX_denom       = 700
        __MY_denom       = height

        self.set_MY(__MY_nom, __MY_denom, debug = False) # meters per pixel in y dimension
        self.set_MX(__MX_nom, __MX_denom, debug = False) # meters per pixel in x dimension
        
        # was the line detected in the last iteration?
        self.detected  = False  
        self.ttlAccepted = 0
        self.ttlRejected = 0
        self.ttlRejectedFramesSinceDetected = 0 
        self.ttlAcceptedFramesSinceRejected = 0 

        self.x_base    = deque([0],  8) 
        self.y_src_bot = y_src_bot
        self.y_src_top = y_src_top
        
        # y values correspoinding to current fitting/ image height
        self.set_ploty()
        self.y_checkpoints = np.concatenate((np.arange(self.y_src_bot,-1,-100), [self.y_src_top]))
        self.y_checkpoints = np.flip(np.sort(self.y_checkpoints))

        # self.curr_RSE            = 0 
        # self.curr_RSE_history    = deque([], 10)
        # self.curr_avg_RSE        = 0
        # self.curr_prev_avg_RSE   = 0
        # self.curr_fit_diffs        = None ## np.array([0,0,0], dtype='float') 
        # self.curr_fit_diffs_history = deque([], self.history)
        # self.best_fit_diffs       = None
        # self.best_fit_diffs_history = deque([], self.history)
        # self.LSE                = 0
        # self.avg_LSE            = 0 
        # self.LSE_history        = []

        #-----------------------------------------------------------------------
        # polynomial coefficients for the most recent fit / fit history
        # fitted_current: x values of the most recent fitting of the line
        # fitted_history: x values of the last n fits of the line  
        #-----------------------------------------------------------------------
        self.curr_fit             = None ## np.array([0,0,0], dtype='float')        
        self.curr_fit_history     = deque([], self.history)

        self.fitted_current       = None        
        self.fitted_history       = deque([], self.history)

        self.compute_best_fit     = deque([], self.history)
        #-----------------------------------------------------------------------
        # Best fit polynomial coefficients, diff with current fitted and history
        #-----------------------------------------------------------------------
        self.best_fit             = None
        self.best_fit_history     = deque([], self.history)
        
        self.fitted_best          = None  
        self.fitted_best_history  = deque([], self.history)

        #-----------------------------------------------------------------------
        ## Various measures of error 
        ## Least squares error between fitted polynom and best_fit
        #-----------------------------------------------------------------------
        self.RSE                = 0  
        self.avg_RSE            = 0
        self.prev_avg_RSE       = 0
        self.RSE_history        = [] 


        self.RMSE               = 0
        self.avg_RMSE           = 0 
        self.RMSE_history       = []

        self.RSE_threshold_history = []

        #radius of curvature of the line in some units
        self.radius_history     = []
        self.radius_avg         = 0.0
        
        #distance in meters of vehicle center from the line
        self.line_base_pixels   = [] 
        self.line_base_meters   = [] 
        self.pixelRatio         = [] 
        self.pixelRatioAvg      = 0

        #slope of detected lane 
        self.slope              = []
        
        # x/y values for detected line pixels
        self.allx = None  
        self.ally = None  
        
        # print(' Line init() complete ')

    def set_MX(self, nom = 3.7, denom = 700, debug = False):
        self.MX_nom   = nom
        self.MX_denom = denom
        self.MX = self.MX_nom/self.MX_denom
        if debug:
            print(' MX            : {} '.format(self.MX))
            print(' MX_nominator  : {} '.format(self.MX_nom))
            print(' MX_denominator: {} '.format(self.MX_denom))
        return self.MX 

        
    def set_MY(self, nom = 30, denom = 720, debug = False):
        self.MY_nom   = nom
        self.MY_denom = denom
        self.MY = self.MY_nom/self.MY_denom
        if debug:
            print(' MY            : {} '.format(self.MY))
            print(' MY_nominator  : {} '.format(self.MY_nom))
            print(' MY_denominator: {} '.format(self.MY_denom))
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
        
    def set_linepos(self, y_eval = 0, poly = None, debug = False):
        lb_pxls, lb_mtrs = self.get_linepos(y_eval, poly, debug = debug)
        self.line_base_pixels.append( lb_pxls )
        self.line_base_meters.append( lb_mtrs )

    def get_linepos(self, y_eval= 0 , poly = None, debug = False):
        if poly is None:
            poly = self.best_fit
        
        x =  np.polyval(poly, y_eval)
        lb_pixels =  round(x,3) 
        lb_meters =  round(x * self.MX,3)

        if debug: 
            print(' get_line_base_pos():  y_eval: {} current_xfitted[y_eval]: {}    MX: {}    lb_meters: {} '.format(y_eval, 
                    x , self.MX, lb_meters))
        return lb_pixels, lb_meters
               

    def set_slope(self, y_eval = 0, fit_parms = None, debug = False):
        slope = self.get_slope(y_eval, fit_parms, debug = debug)
        self.slope.append(slope)

    def get_slope(self, y_eval = 0, fit_parms = None, debug = False):
        if fit_parms is None:
            fit_parms = self.curr_fit
        A,B = fit_parms[0:2] 

        slope  = ((2*A*(y_eval))+B)
        
        slope1  = np.round(np.rad2deg(np.arctan(slope)),3)
        slope2  = slope1 + 90.0
        if debug:
            print(' get_slope(): x: {}  y:{}  slope', slope, ' slope1: ', slope1, ' slope2: ', slope2, ' (deg) - based on pixels')
        return slope2


    def set_radius(self, y_eval = 0 , fit_parms = None, debug = False):
        cur_radius = self.get_radius(y_eval, fit_parms, debug = debug)
        self.radius_history.append( cur_radius )
        self.radius_avg = sum(self.radius_history[-5:])/ len(self.radius_history[-5:]) 
        


    def get_radius(self, y_eval, fit_parms  =None, debug = False):
        if fit_parms is None:
            fit_parms = self.curr_fit
        
        y_eval_MY      = y_eval * self.MY
        exponents      = np.arange(self.POLY_DEGREE,-1,-1)
        MY_factors     = np.power((1.0 / self.MY), exponents)
        fit_parms_mod  = fit_parms * MY_factors * self.MX

        firstDerivParms  = np.polyder(fit_parms_mod, 1)
        firstDeriv_eval  = np.polyval(firstDerivParms, y_eval_MY )
        secondDerivParms = np.polyder(fit_parms_mod, 2)
        secondDeriv_eval = np.polyval(secondDerivParms, y_eval_MY)

        cur_radius = ((1 + (firstDeriv_eval)**2)** 1.5)/np.absolute(secondDeriv_eval) 
        if debug:
            print(' MY              : ', self.MY,  ' MY_inv: ', 1.0 / self.MY)
            print(' y_eval          : ', y_eval   ,  ' y_eval * MY     : ', y_eval_MY)
            print(' expoents        : ', exponents)
            print(' (1/MY)**i       : ', MY_factors)
            print(' fit parms       : ', fit_parms  ,  ' fit_parms_mod : ', fit_parms_mod)
            print(' firstDerivParms : ', type(firstDerivParms),' - ', firstDerivParms)
            print(' firstDeriv_eval : ', type(firstDeriv_eval),' - ', firstDeriv_eval)
            print(' secondDerivParms: ', type(secondDerivParms),' - ', secondDerivParms)
            print(' secondDeriv_eval: ',  type(secondDeriv_eval),' - ',secondDeriv_eval)
            print(' get_radius2()   : ', cur_radius, ' (m)')
        
        cur_radius = np.clip(cur_radius, 0, 6000).tolist()
        
        return np.round(cur_radius,2) 


    def fit_polynomial(self, debug = False):
        # y_range_idxs = (self.ally > self.end >  dst_pts_left[0,:,1]) & ( dst_pts_left[0,:,1] >= start) 
        try:
            self.curr_fit = np.polyfit(self.ally , self.allx , self.POLY_DEGREE, full=False)
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('---------------------------------------------------')
            print('fitPolynomial(): The function failed to fit a line!')
            print(' current_fit will be set to best_fit               ')
            print('---------------------------------------------------')

        self.yfound_min     = self.ally.min()
        self.yfound_max     = self.ally.max()
        self.yrange_found   = np.linspace(self.ally.min(),  self.ally.max(),  self.ally.max()-self.ally.min()+1, dtype = np.int)    
        
        xfitted                = np.polyval(self.curr_fit, self.ploty)
        self.fitted_current    = np.vstack((xfitted, self.ploty))

        self.current_slope   = self.get_slope(self.y_checkpoints,self.curr_fit)
        self.current_radius  = self.get_radius(self.y_checkpoints, self.curr_fit   )
        self.current_linepos = np.polyval(self.curr_fit, self.y_checkpoints)  
        
        if debug:
            print()
            print('\nfit_polynomial: '+self.name+'Lane')
            print('-'*45)
            print('      ally  min: ', self.ally.min(), ' max():',self.ally.max())
            print('   xfitted  min: ', round(xfitted.min(),0), ' max():',round(xfitted.max(),0))
            print('  Proposed poly: ', self.curr_fit)
            print('    best fitted: ', self.best_fit)
            print(' current_radius: ', self.current_radius)
            # print(' xfitted current : ', xfitted.shape, ' - ', xfitted)   
            # print(' self.fitted_current    : ', self.fitted_current.shape)
            # print(' self.current_slope     : ', self.current_slope.shape)
            # print(' self.current_slope_mtrs: ', self.current_slope_mtrs.shape)
            # print(' self.current_linepos   : ', self.current_linepos.shape)


        
    def assessFittedPolynomial(self, debug = False):
        if debug:
            print()
            print('\nAssess Fitted Polynomial for ', self.name, ' Lane:')
            print('-'*45)    

        if self.best_fit is None:  
            self.best_fit = np.copy(self.curr_fit)
            if debug:
                print('  Best fit set to self.curr_fit ... best_fit :', self.best_fit,  ' curr_fit: ', self.curr_fit )

        curr_poly_x = np.polyval(self.curr_fit, self.yrange_found)
        best_poly_x = np.polyval(self.best_fit, self.yrange_found)
        poly_diffs  = (self.best_fit - self.curr_fit)
        
        self.RSE    = np.round(np.sqrt(np.sum(poly_diffs**2)),3)                
        # self.LSE    = np.round(np.sqrt(np.sum((best_poly_x - curr_poly_x)**2)),3)
        self.RMSE   = np.round(np.sqrt(np.sum((best_poly_x - curr_poly_x)**2)/ curr_poly_x.shape[0] ),3)
        
        if debug:
            print('**  ALLY  min : {}  max: {}  yrange_found shape: {}  best_poly_x shape: {}  curr_poly_x shape:{}'.format(
                    self.ally.min(), self.ally.max(), self.yrange_found.shape[0] , best_poly_x.shape[0], curr_poly_x.shape[0] ))
            print('    PixelRatio: {:8.1f}  Avg:{:8.1f}  History: {} '.format(self.pixelRatio[0] , self.pixelRatioAvg , self.pixelRatio[-12:]))
            # print('    LS  Error : {:8.1f}  Avg:{:8.1f}  History: {} '.format(self.LSE , self.avg_LSE , self.LSE_history[-12:]))
            print('    RMS Error : {:8.1f}  Avg:{:8.1f}  History: {} '.format(self.RMSE, self.avg_RMSE, self.RMSE_history[-12:]))
            print('    RSE Error : {:8.1f}  Avg:{:8.1f}  History: {} '.format(self.RSE , self.avg_RSE , self.RSE_history[-12:]))
            print()

        self.RSE_history.append(self.RSE)
        # self.LSE_history.append(self.LSE)
        self.RMSE_history.append(self.RMSE)

        self.prev_avg_RSE = self.avg_RSE
        self.avg_RSE      = np.round(sum(self.RSE_history[-10:])/ len(self.RSE_history[-10:]),3)
        # self.avg_LSE      = np.round(sum(self.LSE_history[-10:])/ len(self.LSE_history[-10:]),3)
        self.avg_RMSE     = np.round(sum(self.RMSE_history[-10:])/len(self.RMSE_history[-10:]),3)
        self.pixelRatioAvg= np.round(sum(self.RSE_history[-10:])/ len(self.RSE_history[-10:]),3) 

        RSE_threshold =  max(self.RSE_THRESHOLD,  self.avg_RSE)
        self.RSE_threshold_history.append(RSE_threshold)
        
        if self.RSE > RSE_threshold :
            if debug:
                print('  > REJECT polynomial - RSE(proposed_fit, best_fit)  >  RSE_THRESHOLD ', self.RSE , ' > ', RSE_threshold)
            rc = False
        else:
            if debug:
                print('  > ACCEPT polynomial - RSE(proposed_fit, best_fit)  <=  RSE_THRESHOLD ', self.RSE , ' <= ', RSE_threshold)
            rc = True
        
        return rc
        
        
    def acceptFittedPolynomial(self, debug= False, debug2 = False):
    
        if debug:
            print()
            print('\nAccept Fitted Polynomial for ', self.name, ' Lane:')
            print('-'*45)    
            print(' Accpeted frames since last rejected : ', self.ttlAcceptedFramesSinceRejected)
            if len(self.curr_fit_history) > 0 :   
                print(' Previous fit      : ', self.curr_fit_history[-1])
            print(' Proposed polyfit  : ', self.curr_fit)
            print(' Current best_fit  : ', self.best_fit)
        
        self.detected = True
        self.ttlAccepted += 1
        self.ttlRejectedFramesSinceDetected = 0 
        self.ttlAcceptedFramesSinceRejected += 1

        # self.set_fit(debug = debug2)
        self.curr_fit_history.append(self.curr_fit)
        self.compute_best_fit.append(self.curr_fit)

        self.set_best_fit(debug = debug2)
        
        self.set_fitted_current()
        self.set_fitted_best()
        
        self.set_lane_stats()
        self.x_base.append(int(self.get_linepos( y_eval= self.y_src_bot)[0]))
        
        if debug: 
            print(' New best_fit      : ', self.best_fit)
            print(' RSE(current, best): {:8.2f}   prev avg_RSE: {:8.2f}    New avg_RSE: {:10.2f} '.format( 
                self.RSE,  self.prev_avg_RSE, self.avg_RSE))
            print(' Old x_base        : {:8d}       New x_base: {:8d} '.format( self.x_base[-2], self.x_base[-1]))


    def rejectFittedPolynomial(self, debug= False, debug2 = False):
    
        self.detected = False
        self.ttlRejected += 1
        self.ttlRejectedFramesSinceDetected += 1 
        self.ttlAcceptedFramesSinceRejected =  0
        
        self.curr_fit = np.copy(self.best_fit)
        
        # self.set_fit(debug = debug2)
        self.curr_fit_history.append(self.curr_fit)
        
        self.set_best_fit(debug = debug2)

        self.set_fitted_current()
        self.set_fitted_best()

        self.set_lane_stats()
        self.x_base.append(int(self.get_linepos( y_eval= self.y_src_bot)[0]))

        if debug:
            print()
            print('\nReject Fitted Polynomial for ', self.name,' Lane:')
            print('-'*45)    
            print(' Consecutive rejected: ', self.ttlRejectedFramesSinceDetected)
            print(' fit history length  : ', len(self.curr_fit_history), '  best_fit_history length   : ', len(self.best_fit_history))
            print(' New Best fit        : ', self.best_fit)
            print(' RSE(current, best)  : {:10.3f}   prev avg_RSE: {:10.3f}    best_avg_RSE: {:10.3f} '.format( 
                self.RSE, self.prev_avg_RSE, self.avg_RSE))
            print(' Old x_base          : {:8d}      New x_base: {:8d} '.format( self.x_base[-2], self.x_base[-1]))

        
            
    def set_lane_stats(self, debug = False):
        self.set_slope(self.y_src_bot, self.best_fit, debug = debug)
        self.set_radius(self.y_src_bot, self.best_fit, debug = debug)
        self.set_linepos(self.y_src_bot, self.best_fit, debug = debug)
        self.best_slope   = self.get_slope(self.y_checkpoints,self.best_fit)
        self.best_radius  = self.get_radius(self.y_checkpoints, self.best_fit)
        self.best_linepos = np.polyval(self.best_fit, self.y_checkpoints) 
            
            
 
    def set_best_fit(self, debug = False):
        ## Calculate new best_fit, using recently added polynomial.

        # self.best_fit  = sum(self.compute_best_fit)/ len(self.compute_best_fit) 
        self.best_fit  = (self.best_fit + self.curr_fit)/2

        self.best_fit_history.append(self.best_fit)

        if debug: 
            print(' set_best_fit()')
            print('   Previous best fit      : ', self.best_fit_history[-2])
            print('   New best fit           : ', self.best_fit_history[-1], '  ' , self.best_fit)
            print('   best_fit_history       : ', len(self.best_fit_history))
            print('   best RSE(current,best) : ', self.RSE, '   best avg_RSE: ', self.prev_avg_RSE, ' new best avg RSE:', self.avg_RSE) 
                

    def set_fitted_current(self, debug = False):
        xfitted = np.polyval(self.curr_fit, self.ploty)
        self.fitted_current   = np.vstack((xfitted, self.ploty))
        self.fitted_history.append(self.fitted_current)        
        
        if debug: 
            print(' set_fitted_current() - length: ', len(self.fitted_history), ' shape: ', self.fitted_current.shape)


    def set_fitted_best(self, debug = False):
        xfitted = np.polyval(self.best_fit, self.ploty)
        self.fitted_best  = np.vstack((xfitted, self.ploty))
        self.fitted_best_history.append(self.fitted_best) 

        if debug: 
            print(' set_best_xfitted() - length: ', len(self.fitted_best_history), ' shape: ', self.fitted_best.shape)


    def reset_best_fit(self, debug = False):
        if debug:
            print(' Clear best fit and history for ', self.name)
        self.best_fit = np.copy(self.curr_fit)
        # compute_best_fit.clear()