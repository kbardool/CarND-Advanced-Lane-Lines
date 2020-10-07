import os,sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import pprint 
from numpy.polynomial import Polynomial  

pp = pprint.PrettyPrinter(indent=2, width=100)
print(' Loading line.py - cwd:', os.getcwd())

# Define a class to receive the characteristics of each line detection
class Line():
    
    def __init__(self, history, height, y_src_top, y_src_bot, **kwargs):

        self.history           = history
        self.compute_history   = kwargs.get('compute_history',2)

        self.name              = kwargs.get('name', '')
        self.POLY_DEGREE       = kwargs.get('poly_degree', 2)
        self.MIN_POLY_DEGREE   = kwargs.get('min_poly_degree', self.POLY_DEGREE)
        self.RSE_THRESHOLD     = kwargs.get('rse_threshold', 120)
        self.MIN_X_SPREAD      = kwargs.get('min_x_spread', 90)
        self.MIN_Y_SPREAD      = kwargs.get('min_y_spread', 350)
        
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
        self.detected    = False  
        self.ttlAccepted = 0
        self.ttlRejected = 0
        self.ttlRejectedFramesSinceDetected = 0 
        self.ttlAcceptedFramesSinceRejected = 0 

        self.y_src_bot   = y_src_bot
        self.y_src_top   = y_src_top
        self.fitPolynomial = None
        # y values correspoinding to current fitting/ image height
        self.set_ploty()
        self.y_checkpoints = np.concatenate((np.arange(self.y_src_bot,-1,-100), [self.y_src_top]))
        self.y_checkpoints = np.flip(np.sort(self.y_checkpoints))

        #-----------------------------------------------------------------------
        # polynomial coefficients for the most recent fit / fit history
        # proposed_curve: x values of the most recent fitting of the line
        # fitted_history: x values of the last n fits of the line  
        #-----------------------------------------------------------------------
        self.proposed_fit         = None ## np.array([0,0,0], dtype='float')        
        self.proposed_fit_history = [np.array([0,0,0], dtype='float') ] 

        self.proposed_curve         = None        
        self.proposed_curve_history = deque([], self.history)

        #-----------------------------------------------------------------------
        # Best fit polynomial coefficients, diff with current fitted and history
        #-----------------------------------------------------------------------
        self.compute_best_fit     = deque([], self.compute_history)
        self.best_fit             = None

        self.best_fit_history     = []
        
        self.fitted_best          = None  
        self.fitted_best_history  = deque([], self.history)

        #-----------------------------------------------------------------------
        ## Various measures of error 
        ## Least squares error between fitted polynom and best_fit
        #-----------------------------------------------------------------------
        self.RSE                 = 0  
        self.avg_RSE             = 0
        self.prev_avg_RSE        = 0
        self.RSE_history         = [] 

        self.RMSE                = 0
        self.avg_RMSE            = 0 
        self.RMSE_history        = []

        self.RSE_threshold_history = []

        #radius of curvature of the line in some units
        self.radius_history      = []
        self.radius_avg          = 0.0
        
        #distance in meters of vehicle center from the line
        self.line_base_pixels    = [] 
        self.line_base_meters    = [] 
        self.pixelRatio          = [] 
        self.pixelCount          = []

        self.avg_pixelRatio      = 0
        self.prev_avg_pixelRatio = 0

        #slope of detected lane 
        self.slope               = []
        self.x_base              = deque([0],  8) 
        
        # x/y values for detected line pixels
        self.allx                = None  
        self.ally                = None  
        self.allx_hist           = []
        self.ally_hist           = []
        self.best_allx           = None  
        self.best_ally           = None  
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
        self.plot_y = np.linspace(start, end-1, end-start,dtype = np.int)
                
    def set_linePixels(self, x_values, y_values):
        self.allx = x_values
        self.ally = y_values
        
        if len(x_values) == 0:
            self.allx_hist.append(( 0, 0))
        else:
            self.allx_hist.append((x_values.min(), x_values.max()))
        
        if len(y_values) == 0:
            self.ally_hist.append(( 0, 0))
        else:
            self.ally_hist.append((y_values.min(), y_values.max()))

    def set_linepos(self, fit_parms = None, y_eval = 0, debug = False):
        lb_pxls, lb_mtrs = self.get_linepos(fit_parms, y_eval, debug = debug)
        self.line_base_pixels.append( lb_pxls )
        self.line_base_meters.append( lb_mtrs )

    def get_linepos(self, fit_parms = None, y_eval = 0 , debug = False):
        if fit_parms is None:
            fit_parms = self.best_fit
        
        x =  np.polyval(fit_parms, y_eval)
        lb_pixels =  round(x,3) 
        lb_meters =  round(x * self.MX,3)

        if debug: 
            print(' get_line_base_pos():  y_eval: {} x[y_eval]: {}    MX: {}    lb_meters: {} '.format(y_eval, 
                    x , self.MX, lb_meters))
        return lb_pixels, lb_meters
               

    def set_slope(self, fit_parms = None, y_eval = 0, debug = False):
        slope = self.get_slope( fit_parms, y_eval, debug = debug)
        self.slope.append(slope)

    def get_slope(self, fit_parms = None, y_eval = 0, debug = False):
        if fit_parms is None:
            fit_parms = self.proposed_fit
        A,B = fit_parms[0:2] 

        slope  = ((2*A*(y_eval))+B)
        
        slope1  = np.round(np.rad2deg(np.arctan(slope)),3)
        slope2  = slope1 + 90.0
        if debug:
            print(' get_slope(): x: {}  y:{}  slope', slope, ' slope1: ', slope1, ' slope2: ', slope2, ' (deg) - based on pixels')
        return slope2


    def set_radius(self, fit_parms = None, y_eval = 0, debug = False):
        cur_radius = self.get_radius( fit_parms, y_eval, debug = debug)
        self.radius_history.append( cur_radius )
        self.radius_avg = sum(self.radius_history[-5:])/ len(self.radius_history[-5:]) 
        

    def get_radius(self, fit_parms = None, y_eval = 0, debug = False):
        
        y_eval_MY      = y_eval * self.MY
        exponents      = np.arange(self.poly_deg,-1,-1)
        MY_factors     = np.power((1.0 / self.MY), exponents)
        
        # print('  fit_parms: {}   exponents: {}   MY_factors: {} '.format(fit_parms, exponents, MY_factors))
        fit_parms_mod  = fit_parms * MY_factors * self.MX

        firstDerivParms  = np.polyder(fit_parms_mod, 1)
        firstDeriv_eval  = np.polyval(firstDerivParms, y_eval_MY )
        secondDerivParms = np.polyder(fit_parms_mod, 2)
        secondDeriv_eval = np.polyval(secondDerivParms, y_eval_MY)

        if np.all(secondDerivParms == np.zeros_like(secondDerivParms)) :
            # print( ' second deriv is zero ')
            cur_radius = np.ones_like(y_eval) * 6000
        else:
            cur_radius = ((1 + (firstDeriv_eval)**2)** 1.5)/np.absolute(secondDeriv_eval) 
        
        if debug:
            print(' MY              : ', self.MY,  ' MY_inv: ', 1.0 / self.MY)
            print(' y_eval          : ', y_eval   ,  ' y_eval * MY     : ', y_eval_MY)
            print(' exponents       : ', exponents)
            print(' (1/MY)**i       : ', MY_factors)
            print(' fit parms       : ', fit_parms  ,  ' fit_parms_mod : ', fit_parms_mod)
            print(' firstDerivParms : ', type(firstDerivParms),' - ', firstDerivParms)
            print(' firstDeriv_eval : ', type(firstDeriv_eval),' - ', firstDeriv_eval)
            print(' secondDerivParms: ', type(secondDerivParms),' - ', secondDerivParms)
            print(' secondDeriv_eval: ', type(secondDeriv_eval),' - ',secondDeriv_eval)
            print(' current radius  : ', cur_radius, ' (m)')
        
        cur_radius = np.clip(cur_radius, 0, 6000).tolist()
        
        return np.round(cur_radius,2) 


    def fit_polynomial(self, debug = False):
        # y_range_idxs = (self.ally > self.end >  dst_pts_left[0,:,1]) & ( dst_pts_left[0,:,1] >= start) 
        
        if debug:
            print()
            print('\nfit_polynomial: '+self.name+' Lane')
            print('-'*45)
            print('  len(allx) {}  len(ally): {} '.format(len(self.allx), len(self.ally)))

        if len(self.allx) < 200:
            self.proposed_fit = np.copy(self.best_fit_history[-1])
            # self.poly_deg = self.best_fit[-1].shape[0] -1
            # print('  *  allx = ally = 0 - using previous best fit --> poly_degree: {} '.format(self.poly_deg))
            self.allx = np.copy(self.best_allx)
            self.ally = np.copy(self.best_ally)
            if debug:
                print('  No pixels detected - revert to previous best_fit polynomial.')
        else:
            x_spread = self.allx.max() - self.allx.min()
            y_spread = self.ally.max() - self.ally.min()

            if x_spread < self.MIN_X_SPREAD and y_spread < self.MIN_Y_SPREAD:
                self.poly_deg = self.MIN_POLY_DEGREE
            else:
                self.poly_deg = self.POLY_DEGREE
            if debug:
                print('  *  {} - x_spread: {}   y_spread: {}    poly_degree: {}  '.format(
                                            self.name, x_spread, y_spread, self.poly_deg ))

            try:
                self.proposed_fit = np.polyfit(self.ally , self.allx , self.poly_deg, full=False)
            except Exception as e:
                # Avoids an error if `left` and `right_fit` are still none or incorrect
                print('--------------------------------------------------------')
                print('  fitPolynomial(): The function failed to fit a line!   ')
                print('  len(allx) {}  len(ally): {} '.format(len(self.allx), len(self.ally)))
                print('  previous best_fit will be used as proposed polynomial ')
                print('--------------------------------------------------------')
                print(' Exception message: ', e)
                self.proposed_fit = np.copy(self.best_fit)
            finally:
                if self.poly_deg ==1:
                    self.proposed_fit = np.concatenate(([0.0], self.proposed_fit))
                    self.poly_deg = 2
                    if debug:
                        print(' **  x_spread: {}   y_spread: {}    poly_degree reset to : {} '.format(x_spread, y_spread, self.poly_deg))

        allx_min, allx_max = self.allx.min(), self.allx.max()
        ally_min, ally_max = self.ally.min(), self.ally.max()
       
        plot_x = np.round(np.polyval(self.proposed_fit, self.plot_y),0).astype(np.int)
        
        self.yrange_found  = np.linspace(ally_min,  ally_max,  ally_max - ally_min+1, dtype = np.int)
        
        self.proposed_fit_history.append(self.proposed_fit)
        self.current_slope   = self.get_slope(self.proposed_fit, self.y_checkpoints)
        self.current_radius  = self.get_radius(self.proposed_fit, self.y_checkpoints)
        self.current_linepos = np.polyval(self.proposed_fit, self.y_checkpoints)  
        
        self.pixel_spread_x = allx_max - allx_min + 1
        self.pixel_spread_y = ally_max - ally_min + 1
        self.curve_spread_x = plot_x.max() - plot_x.min()+1

        if debug:
            print('  X points(allx)   min : {:6d}   max:  {:6d}  spread: {:6d}'.format(allx_min, allx_max, self.pixel_spread_x ))
            print('  Y points(ally)   min : {:6d}   max:  {:6d}  spread: {:6d}'.format(ally_min, ally_max, self.pixel_spread_y ))
            print('  Fitted curve X   min : {:6d}   max:  {:6d}  spread: {:6d}'.format(plot_x.min(), plot_x.max(), self.curve_spread_x))
            print('  Current best_fit     : ', self.best_fit)
            print('  Proposed polynomial  : ', self.proposed_fit)
            print('  Current radius       : ', self.current_radius)

        self.assessFittedPolynomial(debug=debug)        

    def assessFittedPolynomial(self, debug = False):
        # if debug:
            # print()
            # print('\n  fitted Polynomials error computation-  '+self.name+' Lane')
            # print(' ','-'*45)    

        if self.best_fit is None:  
            self.best_fit = np.copy(self.proposed_fit)
            if debug:
                print('  Best fit set to self.curr_fit ... best_fit :', self.best_fit,  ' proposed_fit: ', self.proposed_fit )

        curr_poly_x = np.polyval(self.proposed_fit, self.yrange_found)
        best_poly_x = np.polyval(self.best_fit, self.yrange_found)
        
        if self.best_fit.shape[0] == self.proposed_fit.shape[0]:  
            poly_diffs  = (self.best_fit - self.proposed_fit)
        else:
            fit_len = min(self.proposed_fit.shape[0], self.best_fit.shape[0])
            poly_diffs  = (self.best_fit[-fit_len:] - self.proposed_fit[-fit_len:])
        
        # print(' curr_poly_x ', curr_poly_x)
        # print(' best_poly_x ', best_poly_x)
      
        self.RSE    = np.round(np.sqrt(np.sum(poly_diffs**2)),3)                
        self.RMSE   = np.round(np.sqrt(np.sum((best_poly_x - curr_poly_x)**2)/ curr_poly_x.shape[0] ),3)
        # self.LSE    = np.round(np.sqrt(np.sum((best_poly_x - curr_poly_x)**2)),3)
        
        
        self.RSE_history.append(self.RSE)
        self.RMSE_history.append(self.RMSE)
        # self.LSE_history.append(self.LSE)

        self.prev_avg_pixelRatio = self.avg_pixelRatio
        self.prev_avg_RSE = self.avg_RSE
        self.prev_avg_RMSE = self.avg_RMSE

        self.avg_pixelRatio= np.round(sum(self.pixelRatio[-10:])/ len(self.pixelRatio[-10:]),3) 
        self.avg_RSE      = np.round(sum(self.RSE_history[-10:])/ len(self.RSE_history[-10:]),3)
        self.avg_RMSE     = np.round(sum(self.RMSE_history[-10:])/len(self.RMSE_history[-10:]),3)
        # self.avg_LSE      = np.round(sum(self.LSE_history[-10:])/ len(self.LSE_history[-10:]),3)

        # RSE_threshold =  min(self.RSE_THRESHOLD,  self.avg_RSE)
        # self.RSE_threshold_history.append(RSE_threshold)
        
        RSE_threshold = self.RSE_THRESHOLD
        self.RSE_threshold_history.append(RSE_threshold)
        
        # msg1 ='  > RSE(proposed_fit, best_fit)  {}  -  RSE_THRESHOLD {}'.format(Lane.RSE, RSE_threshold) 
        # if (Lane.RSE < RSE_threshold):    ## or (self.acceptHistory[-1] < 0):
            # Lane.acceptPolynomial =  True
        # else:
            # Lane.acceptPolynomial =  False
            # msg1 = '  > REJECT polynomial - RSE(proposed_fit, best_fit)  {}  >   RSE_THRESHOLD {} '.format(Lane.RSE, RSE_threshold)
            
        if debug:
            
            print('  Pxl Ratio : {:7.1f}  Prev Avg: {:7.1f}  Avg:{:7.1f}  H: {} '.format(self.pixelRatio[-1], self.prev_avg_pixelRatio , 
                     self.avg_pixelRatio , self.pixelRatio[-12:]))
            print('  RMS Error : {:7.1f}  Prev Avg: {:7.1f}  Avg:{:7.1f}  H: {} '.format(self.RMSE, self.prev_avg_RMSE, self.avg_RMSE, 
                     self.RMSE_history[-11:]))
            print('  RSE Error : {:7.1f}  Prev Avg: {:7.1f}  Avg:{:7.1f}  H: {} '.format(self.RSE , self.prev_avg_RSE , self.avg_RSE , 
                     self.RSE_history[-11:]))
            # print('  LS  Error : {:8.1f}  Avg:{:8.1f}  History: {} '.format(self.LSE , self.avg_LSE , self.LSE_history[-12:]))
            print('  RSE(proposed_fit, best_fit)  {:7.1f}  -  RSE_THRESHOLD {:7.1f}'.format(self.RSE, self.RSE_THRESHOLD))
        return True
        
        
    def acceptFittedPolynomial(self, debug= False, debug2 = False):
    
        if debug:
            print()
            print('Accept Fitted Polynomial for '+self.name+' Lane')
            print('-'*45)    
            print(' Accpeted frames since last rejected : ', self.ttlAcceptedFramesSinceRejected)
            # if len(self.proposed_fit_history) > 0 :   
                # print(' Previous fit      : ', self.proposed_fit_history[-1])
            print(' Proposed fit[-2]      : ', self.proposed_fit_history[-2])
            print(' Current  Proposed fit : ', self.proposed_fit_history[-1])
            print(' Current  best_fit     : ', self.best_fit)
        
        self.detected = True
        self.ttlAccepted += 1
        self.ttlRejectedFramesSinceDetected = 0 
        self.ttlAcceptedFramesSinceRejected += 1

        self.set_best_fit(debug = debug2)
        self.set_proposed_curve()
        self.set_fitted_best()
        
        self.set_lane_stats()
        self.x_base.append(int(self.get_linepos( y_eval= self.y_src_bot)[0]))
        
        if debug: 
            print(' New best_fit          : ', self.best_fit)
            print(' RSE(current, best)    : {:8.2f}   prev avg_RSE: {:8.2f}    New avg_RSE: {:10.2f} '.format( 
                self.RSE,  self.prev_avg_RSE, self.avg_RSE))
            print(' Old x_base            : {:8d}       New x_base: {:8d} '.format( self.x_base[-2], self.x_base[-1]))


    def rejectFittedPolynomial(self, debug= False, debug2 = False):
        if debug:
            print()
            print('Reject Fitted Polynomial for '+self.name+' Lane')
            print('-'*45)    
            print(' Rejected frames since last accepted : ', self.ttlRejectedFramesSinceDetected)
            # if len(self.proposed_fit_history) > 0 :   
                # print(' Previous fit      : ', self.proposed_fit_history[-1])
            print(' Proposed fit[-2]      : ', self.proposed_fit_history[-2])
            print(' Current  Proposed fit : ', self.proposed_fit_history[-1])
            print(' Current  best_fit     : ', self.best_fit)
    
        self.detected = False
        self.ttlRejected += 1
        self.ttlRejectedFramesSinceDetected += 1 
        self.ttlAcceptedFramesSinceRejected =  0
        
        self.proposed_fit = np.copy(self.best_fit)
        # self.poly_deg = self.best_fit.shape[0] - 1

        self.set_best_fit(debug = debug2)
        self.set_proposed_curve()
        self.set_fitted_best()

        self.set_lane_stats()
        self.x_base.append(int(self.get_linepos( y_eval= self.y_src_bot)[0]))

        if debug: 
            print(' New best_fit          : ', self.best_fit)
            print(' RSE(current, best)    : {:8.2f}   prev avg_RSE: {:8.2f}    New avg_RSE: {:10.2f} '.format( 
                self.RSE,  self.prev_avg_RSE, self.avg_RSE))
            print(' Old x_base            : {:8d}       New x_base: {:8d} '.format( self.x_base[-2], self.x_base[-1]))

            
    def set_lane_stats(self, debug = False):
        self.set_slope(self.best_fit, self.y_src_bot, debug = debug)
        self.set_radius(self.best_fit, self.y_src_bot, debug = debug)
        self.set_linepos(self.best_fit, self.y_src_bot, debug = debug)
        self.best_slope   = self.get_slope(self.best_fit, self.y_checkpoints,)
        self.best_radius  = self.get_radius(self.best_fit, self.y_checkpoints,)
        self.best_linepos = np.polyval(self.best_fit, self.y_checkpoints) 
            
            
    def set_best_fit(self, debug = False):
        ## Calculate new best_fit, using recently added polynomial.

        self.compute_best_fit.append(self.proposed_fit)
        self.best_fit  = sum(self.compute_best_fit)/ len(self.compute_best_fit) 
        
        self.best_fit_history.append(self.best_fit)
        self.best_allx = np.copy(self.allx)
        self.best_ally = np.copy(self.ally)

        if debug: 
            print(' set_best_fit()')
            print('   Previous best fit      : ', self.best_fit_history[-2] if len(self.best_fit_history) > 1 else 'N/A')
            print('   New best fit           : ', self.best_fit_history[-1], '  ' , self.best_fit)
            print('   best_fit_history       : ', len(self.best_fit_history))
            print('   best RSE(current,best) : ', self.RSE, '   best avg_RSE: ', self.prev_avg_RSE, ' new best avg RSE:', self.avg_RSE) 
                

    def set_proposed_curve(self, debug = False):
        x = np.polyval(self.proposed_fit, self.plot_y)
        self.proposed_curve   = np.vstack((x, self.plot_y))
        self.proposed_curve_history.append(self.proposed_curve)        
        
        if debug: 
            print(' set_proposed_curve() - length: ', len(self.proposed_curve_history), ' shape: ', self.proposed_curve.shape)


    def set_fitted_best(self, debug = False):
        x = np.polyval(self.best_fit, self.plot_y)
        self.fitted_best  = np.vstack((x, self.plot_y))
        self.fitted_best_history.append(self.fitted_best) 

        if debug: 
            print(' set_fitted_best() - length: ', len(self.fitted_best_history), ' shape: ', self.fitted_best.shape)


    def reset_best_fit(self, debug = False):
        if debug:
            print('    Clear best fit and history for ', self.name)
        self.best_fit = np.copy(self.proposed_fit)