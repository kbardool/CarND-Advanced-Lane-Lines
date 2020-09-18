import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def plot_1(pl, legend = 'best', size=(15,7)):
    plt.figure(figsize=size)
    ax = plt.gca()
    min_x = min(pl.imgUndistStats['RGB'])
    max_x = max(pl.imgUndistStats['RGB'])
    len_x = len(pl.imgUndistStats['RGB'])
    # ax.plot(pl.offctr_history    , label = 'Off Center') ### ,  color=SCORE_COLORS[score_key])
    ax.plot(pl.imgPixelRatio     ,color='brown', label = 'Pixel Rto')
    ax.plot(pl.imgUndistStats['RGB'] , label = 'Undist RGB Mean')

    ax.plot(np.array(pl.imgAcceptHistory), label = 'Polynom Acpt/Rjct')
    ax.plot(np.array(pl.imgThrshldHistory), label = 'ImgCondition')
    ax.set_title('Plot 1 - Pxl Thrshld: {:4.2f} OffCt Thrshld: {:3d}'.format( pl.IMAGE_RATIO_HIGH_THRESHOLD, pl.CURRENT_OFFCTR_ROI_THR ))
    plt.hlines( pl.HIGH_RGB_THRESHOLD      , 0, len_x, color='red' , alpha=0.5, linestyles='dashed', linewidth=1, label = 'HI RGB')    
    plt.hlines( pl.MED_RGB_THRESHOLD       , 0, len_x, color='green' , alpha=0.5, linestyles='dashed', linewidth=1, label = 'MED RGB')    
    plt.hlines( pl.LOW_RGB_THRESHOLD       , 0, len_x, color='yellow' , alpha=0.5, linestyles='dashed', linewidth=1, label = 'LOW RGB')    
    plt.hlines( pl.IMAGE_RATIO_HIGH_THRESHOLD, 0, len_x, color='red'  , alpha=0.5, linestyles='dashed', linewidth=1, label='PxlRatioThr')    
    plt.hlines( pl.OFF_CENTER_ROI_THRESHOLD, 0, len_x, color='black', alpha=0.8, linestyles='dashed', linewidth=1, label='OffCtrRoIThr' )    
    # plt.hlines(-pl.CURRENT_OFFCTR_ROI_THR, 0, len_x, color='black', alpha=0.8, linestyles='dashed', linewidth=1)    
    
    for (x,_,_) in pl.imgAdjustHistory:
        ax.plot(x,-40, 'bo') 
    leg = plt.legend(loc=legend,frameon=True, fontsize = 10, markerscale = 6)
    leg.set_title('Legend',prop={'size':11})


def plot_2(pl, Lane, ttl = '' ,legend = 'best', size=(15,7)):
    plt.figure(figsize=size)
    ax = plt.gca()
    min_x = min(pl.imgUndistStats['RGB'])
    max_x = max(Lane.RSE_history)
    len_x = len(pl.imgUndistStats['RGB'])
    ax.plot(Lane.pixelRatio, label = 'Pixel Rto')
    ax.plot(np.array(Lane.RSE_history), label = 'RSE Error')
    ax.plot(np.array(pl.imgAcceptHistory), label = 'Polynom Acpt/Rjct')
    
    plt.axhline( -20, 0, 1, color='black', alpha=0.9, ls='dotted', lw=1)  
    plt.hlines( 80 , 0, len_x, color='green' , alpha=0.8, linestyles='dashed', linewidth=1, label = '<80>')    
    plt.hlines( 50 , 0, len_x, color='blue'  , alpha=0.8, linestyles='dashed', linewidth=1, label = '<50>')    
    plt.hlines( 15 , 0, len_x, color='maroon', alpha=0.8, linestyles='dashed', linewidth=1, label = '<15>')    
    # plt.hlines( pl.CURRENT_OFFCTR_ROI_THR, 0, len_x, color='black', alpha=1.0, linestyles='dashed', linewidth=1)    
    # plt.hlines(-pl.CURRENT_OFFCTR_ROI_THR, 0, len_x, color='black', alpha=1.0, linestyles='dashed', linewidth=1)    

    for (x,_,_) in pl.imgAdjustHistory:
        ax.plot(x,-20, 'bo') 
    plt.ylim(-25, max_x + 5)

    ax.set_title(' Plot 2 - '+Lane.name+ ' Lane - Pixel Threshld: {:3d}       OffCtr Thrshld: {:3d}'.format( 
                    pl.IMAGE_RATIO_HIGH_THRESHOLD, pl.CURRENT_OFFCTR_ROI_THR ))
    leg = plt.legend(loc=legend,frameon=True, fontsize = 12,markerscale = 6)
    leg.set_title('Legend',prop={'size':11})
                                    


def plot_3(pl, legend = 'best', size=(15,7), pxlthr = False):
    plt.figure(figsize=size)
    ax = plt.gca()
    len_x = len(pl.imgUndistStats['RGB'])

    ax.plot(pl.imgWarpedStats['Hue'], color='r', label='Hue (Warped)')
    ax.plot(pl.imgWarpedStats['Sat'], color='b', label='Sat (Warped)')
    ax.plot(pl.imgWarpedStats['Lvl'], color='g', label='Lvl/RGB (Warped)')
    # ax.plot(pl.imgWarpedStats['RGB'], color='darkorange', label='RGB (Warped)')
    # ax.plot(pl.imgUndistStats['Sat'], color='darkblue', alpha = 0.5, label='Sat')
    # ax.plot(pl.imgUndistStats['RGB'], color='darkorange', alpha = 0.5,   label='RGB ')
    # ax.plot(pl.imgUndistStats['Red'], color='r', label='Red ')
    # ax.plot(pl.imgUndistStats['Grn'], color='g', label='Grn ')
    # ax.plot(pl.imgUndistStats['Blu'], color='b', label='Blu ')
    # ax.plot(pl.imgUndistStats['Hue'], color='r', label='Hue')
    # ax.plot(pl.imgWarpedStats['Red'], color='r', label='Red (Warped)', linestyle='dashed')
    # ax.plot(pl.imgWarpedStats['Grn'], color='g', label='Grn (Warped)', linestyle='dashed')
    # ax.plot(pl.imgWarpedStats['Blu'], color='b', label='Blu (Warped)', linestyle='dashed')
    # ax.plot(pl.imgWarpedStats['Hue'], color='r', label='Hue (Warped)', linestyle='dashed')
    # plt.axhline( -20, 0, 1, color='black', alpha=0.9, ls='dotted', lw=1)  
    
    # ax.plot(np.array(pl.imgThrshldHistory), label = 'Normal/Dark')
    if pxlthr:
        ax.plot(pl.imgPixelRatio, color='sienna', label = 'Pixel Rto')

    # plt.hlines( 180      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = '(180)')    
    # plt.hlines( 150      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = '(150)')    
    plt.hlines( pl.HIGH_RGB_THRESHOLD    , 0, len_x, color='darkred', alpha=0.5, linestyles='dashed', linewidth=1, 
                label = 'High RGB '+str(pl.HIGH_RGB_THRESHOLD))    
    plt.text(0, pl.HIGH_RGB_THRESHOLD - 30, 'HighRGB', family = 'monospaced')

    plt.hlines( pl.MED_RGB_THRESHOLD    , 0, len_x, color='darkorange', alpha=0.7, linestyles='dashed', linewidth=1, 
                label = 'Med RGB '+str(pl.MED_RGB_THRESHOLD))    
    plt.text(0, pl.MED_RGB_THRESHOLD - 30, 'MedRGB', family = 'monospaced')
    plt.axhspan(pl.MED_RGB_THRESHOLD, pl.LOW_RGB_THRESHOLD,facecolor='g', alpha = 0.5)
    
    plt.hlines( pl.LOW_RGB_THRESHOLD    , 0, len_x, color='darkgreen' , alpha=0.5, linestyles='dashed', linewidth=1,
                label = 'Low RGB '+str(pl.LOW_RGB_THRESHOLD)) 
    plt.text(0, pl.LOW_RGB_THRESHOLD - 30, 'LowRGB', family = 'monospaced')
    plt.axhspan(pl.LOW_RGB_THRESHOLD, pl.VLOW_RGB_THRESHOLD,facecolor='y', alpha = 0.5)
    
    plt.hlines( pl.VLOW_RGB_THRESHOLD, 0, len_x, color='darkred'      , alpha=0.5, linestyles='dashed', linewidth=1,  
                label = 'VLow RGB '+str(pl.VLOW_RGB_THRESHOLD))    
    plt.text(0, pl.VLOW_RGB_THRESHOLD - 30, 'VLowRGB', family = 'monospaced')
    plt.axhspan(pl.VLOW_RGB_THRESHOLD, 0 , facecolor='mistyrose', alpha = 0.5)
    
    # plt.hlines( pl.IMAGE_RATIO_HIGH_THRESHOLD, 0, len_x, color='black'  , alpha=0.5, linestyles='dashed', linewidth=1)    
    
    ax.minorticks_on()        
    for (x,_,_) in pl.imgAdjustHistory:
        ax.plot(x,-20, 'bo') 

    leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
    leg.set_title('Legend',prop={'size':11})
    ax.set_title('Plot 3 - WARPED Image RGB Avgs - Pxl Thrshld: {:4.2f} OffCt Thrshld: {:3d}'.format( 
                    pl.IMAGE_RATIO_HIGH_THRESHOLD, pl.CURRENT_OFFCTR_ROI_THR ))


def plot_4U(pl, legend = 'best', size=(15,7), pxlthr=False, start=0, vmark  = None):
    plt.figure(figsize=size)
    ax = plt.gca()
    len_x = len(pl.imgUndistStats['Hue'][start:])

    ax.plot(pl.imgUndistStats['Hue'][start:], color='red'  , label='Hue')
    ax.plot(pl.imgUndistStats['Lvl'][start:], color='green', label='Lvl')
    ax.plot(pl.imgUndistStats['Sat'][start:], color='blue' , label='Sat')
    ax.plot(pl.imgUndistStats['RGB'][start:], color='darkorange', label='RGB')
    ax.plot(np.array(pl.imgThrshldHistory[start:]), label = 'ImgCondition')
    # ax.plot(pl.imgUndistStats['HLS'] ,color='k', label='HLS')
    ax.plot(pl.diffsSrcDynPoints, color='black', label='diff')
    # ax.plot(np.array(pl.imgAcceptHistory) , label = 'Polynom Acpt/Rjct')
    if pxlthr:
        ax.plot(pl.imgPixelRatio[start:], color='sienna', label = 'Pixel Rto')

    plt.hlines( pl.HIGH_RGB_THRESHOLD, 0, len_x, color='darkred'  , alpha=0.5, linestyles='dashed', linewidth=1, 
                label = 'High RGB '+str(pl.HIGH_RGB_THRESHOLD))    
    plt.hlines( pl.MED_RGB_THRESHOLD , 0, len_x, color='darkgreen', alpha=0.5, linestyles='dashed', linewidth=1, 
                label = 'Med RGB '+str(pl.MED_RGB_THRESHOLD))    
    plt.hlines( pl.LOW_RGB_THRESHOLD , 0, len_x, color='brown'    , alpha=0.5, linestyles='dashed', linewidth=1,  
                label = 'Low RGB '+str(pl.LOW_RGB_THRESHOLD))    
    plt.hlines( pl.VLOW_RGB_THRESHOLD, 0, len_x, color='red'      , alpha=0.5, linestyles='dashed', linewidth=1,  
                label = 'VLow RGB '+str(pl.VLOW_RGB_THRESHOLD))    

    plt.axhspan(pl.MED_RGB_THRESHOLD, pl.LOW_RGB_THRESHOLD,facecolor='g', alpha = 0.5)
    plt.axhspan(pl.LOW_RGB_THRESHOLD, pl.VLOW_RGB_THRESHOLD,facecolor='y', alpha = 0.5)
    plt.axhspan(pl.VLOW_RGB_THRESHOLD, 0 , facecolor='lightcoral', alpha = 0.5)

    if vmark:  
        plt.axvline(vmark - start, 0, 1, color='red'  , alpha=0.5, ls='dashed', lw=1)    

    # plt.hlines( pl.LOW_SAT_THRESHOLD , 0, len_x, color='g'   , alpha=0.5, linestyles='dashed', linewidth=1,
    #             label = 'High SAT '+str(pl.HIGH_SAT_THRESHOLD))    
    # plt.hlines( pl.HIGH_SAT_THRESHOLD, 0, len_x, color='k'   , alpha=0.5, linestyles='dashed', linewidth=1, 
    #             label = 'Low SAT '+str(pl.HIGH_SAT_THRESHOLD))    
    # plt.hlines( 200      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
    # plt.hlines( 150      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
    
    for (x,_,_) in pl.imgAdjustHistory[start:]:
        ax.plot(x,-20, 'bo') 

    ax.minorticks_on()
    leg = plt.legend(loc=legend,frameon=True, fontsize = 10, markerscale = 6)
    leg.set_title('Legend',prop={'size':11})
    ax.set_title('Plot 4U - UNDIST - Pxl Thrshld: {:5.1f} OffCt Thrshld: {:3d}'.format( 
                    pl.IMAGE_RATIO_LOW_THRESHOLD, pl.CURRENT_OFFCTR_ROI_THR ))


def plot_4W(pl, legend = 'best', size=(15,7), pxlthr=False, start = 0, vmark = None):
    plt.figure(figsize=size)
    ax = plt.gca()

    len_x = len(pl.imgUndistStats['Hue'][start:])
    ax.plot(pl.imgWarpedStats['Hue'][start:], color='red'  , label='Hue (W)')
    ax.plot(pl.imgWarpedStats['Lvl'][start:], color='green', label='Lvl (W)')
    ax.plot(pl.imgWarpedStats['Sat'][start:], color='blue' , label='Sat (W)')
    ax.plot(pl.imgWarpedStats['RGB'][start:], color='darkorange', label='RGB (W)')
    # ax.plot(pl.imgUndistStats['RGB'][start:], color='darkblue'  , alpha = 0.2, label='RGB (U)', linestyle='dashed')
    # ax.plot(pl.imgWarpedStats['HLS']   , color='k', label='HLS')
    # ax.plot(np.array(pl.imgAcceptHistory) , label = 'Polynom Acpt/Rjct')
    
    ax.plot(np.array(pl.imgThrshldHistory[start:]), label = 'ImgCondition')
    if pxlthr:
        ax.plot(pl.imgPixelRatio[start:], color='sienna', label = 'Pixel Rto')

    plt.hlines( pl.HIGH_RGB_THRESHOLD, 0, len_x, color='darkred'  , alpha=0.5, linestyles='dashed', linewidth=1, 
                label = 'High RGB '+str(pl.HIGH_RGB_THRESHOLD))    
    plt.hlines( pl.MED_RGB_THRESHOLD , 0, len_x, color='darkgreen', alpha=0.5, linestyles='dashed', linewidth=1, 
                label = 'Med RGB '+str(pl.MED_RGB_THRESHOLD))    
    plt.hlines( pl.LOW_RGB_THRESHOLD , 0, len_x, color='brown'    , alpha=0.5, linestyles='dashed', linewidth=1,  
                label = 'Low RGB '+str(pl.LOW_RGB_THRESHOLD))       

    plt.hlines( pl.VLOW_RGB_THRESHOLD, 0, len_x, color='red'      , alpha=0.5, linestyles='dashed', linewidth=1,  
                label = 'VLow RGB '+str(pl.VLOW_RGB_THRESHOLD))       
    plt.hlines( pl.LOW_SAT_THRESHOLD , 0, len_x, color='black'    , alpha=0.5, linestyles='dotted', linewidth=1,
                label = 'Low SAT '+str(pl.LOW_SAT_THRESHOLD))       
    plt.hlines( pl.HIGH_SAT_THRESHOLD, 0, len_x, color='black'    , alpha=0.5, linestyles='dotted', linewidth=1, 
                label = 'Hi SAT '+str(pl.HIGH_SAT_THRESHOLD))    
    # plt.axhline( 85, 0, 1, color='black', alpha=0.9, ls='dotted', lw=1)  

    plt.axhspan(pl.MED_RGB_THRESHOLD, pl.LOW_RGB_THRESHOLD,facecolor='g', alpha = 0.5)
    plt.axhspan(pl.LOW_RGB_THRESHOLD, pl.VLOW_RGB_THRESHOLD,facecolor='y', alpha = 0.5)
    plt.axhspan(pl.VLOW_RGB_THRESHOLD, 0 , facecolor='mistyrose', alpha = 0.5)

    if vmark:  
        plt.axvline(vmark - start, 0, 1, color='red'  , alpha=0.5, ls='dashed', lw=1)    

    # plt.hlines( 175      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
    # plt.hlines( 120      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
    # plt.hlines( 200      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
    # plt.hlines( 150      , 0, len_x, color='k', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
    ax.minorticks_on()        
    for (x,_,_) in pl.imgAdjustHistory[start:]:
        ax.plot(x,-10, 'bo') 

    leg = plt.legend(loc=legend,frameon=True, fontsize = 10, markerscale = 6)
    leg.set_title('Legend',prop={'size':11})
    ax.set_title('Plot 4W - WARPED - Pxl Thrshlds: {:5.1f} OffCt Thrshld: {:3d}'.format(
                    pl.IMAGE_RATIO_LOW_THRESHOLD, pl.CURRENT_OFFCTR_ROI_THR ))



def plot_5(pl, legend = 'best', size=(15,7)):
    plt.figure(figsize=size)
    ax = plt.gca()
    len_x = len(pl.imgUndistStats['RGB'])
    ax.plot(pl.LeftLane.pixelRatio , color = 'r', label='Left Pxl Ratio') 
    ax.plot(pl.RightLane.pixelRatio, color = 'b', label='Right Pxl Ratio') 
    ax.plot(pl.imgPixelRatio       , color = 'k', label='Img Pxl Ratio') 
    ax.plot(pl.imgWarpedStats['Sat'], label='Sat', linestyle='dashed'  )


    # ax.plot(np.array(pl.imgAcceptHistory) , label = 'Polynom Acpt/Rjct')
    # ax.plot(np.array(pl.imgThrshldHistory), label = 'Normal/Dark')

    ax.set_title('Plot 5 - Pxl Thrshld: {:5.1f} OffCt Thrshld: {:3d}'.format( pl.IMAGE_RATIO_LOW_THRESHOLD, pl.CURRENT_OFFCTR_ROI_THR ))
    plt.hlines( pl.LANE_RATIO_LOW_THRESHOLD  , 0, len_x, color='blue' , alpha=0.5, linestyles='dashed', linewidth=1)    
    plt.hlines( pl.LANE_RATIO_HIGH_THRESHOLD , 0, len_x, color='blue' , alpha=0.5, linestyles='dashed', linewidth=1)    
    plt.hlines( pl.IMAGE_RATIO_HIGH_THRESHOLD, 0, len_x, color='red'  , alpha=0.8, linestyles='dashed', linewidth=1, label='<30>')    
    plt.hlines( pl.IMAGE_RATIO_LOW_THRESHOLD, 0, len_x, color='red'  , alpha=0.8, linestyles='dashed', linewidth=1, label='<30>')    
    
    plt.text(0, pl.LANE_RATIO_HIGH_THRESHOLD + 2, 'Lane Ratio Upper Threshold', family = 'monospaced')            
    plt.text(0, pl.IMAGE_RATIO_HIGH_THRESHOLD - 2, 'Image Ratio Upper Threshold', family = 'monospaced')            
    plt.text(0, pl.LANE_RATIO_LOW_THRESHOLD - 2, 'Lane Ratio Lower Threshold', family = 'monospaced')            
    # plt.hlines( pl.MED_RGB_THRESHOLD  , 0, len_x, color='darkgreen', alpha=0.5, linestyles='dashed', linewidth=1, label = 'MedRGB')    
    # plt.hlines( pl.CURRENT_OFFCTR_ROI_THR, 0, len_x, color='black', alpha=0.8, linestyles='dashed', linewidth=1)    
    # plt.hlines(-pl.CURRENT_OFFCTR_ROI_THR, 0, len_x, color='black', alpha=0.8, linestyles='dashed', linewidth=1)    
    
    for (x,_,_) in pl.imgAdjustHistory:
        ax.plot(x, -10, 'bo') 
    leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
    leg.set_title('Legend',prop={'size':11})



def plot_6(pl, legend = 'best', size=(15,7), clip = 9999999):
    plt.figure(figsize=size)
    ax = plt.gca()
    len_x = len(pl.imgUndistStats['RGB'])
    ax.plot(np.array( pl.LeftLane.RSE_history), label = 'Left RSE ')
    ax.plot( np.clip( pl.LeftLane.radius_history ,0, clip), color = 'r', label='Left Radius') 
    ax.plot( np.clip( pl.RightLane.radius_history,0, clip), color = 'b', label='Right Radius') 
    # ax.plot(np.array(pl.LeftLane.radius_history)  , color = 'r', label='Left Radius') 
    # ax.plot(np.array(pl.RightLane.radius_history) , color = 'b', label='Right Radius') 
    
    plt.hlines( 10  , 0, len_x, color='red'  , alpha=0.8, linestyles='dashed', linewidth=1, label='<10>')    
    plt.hlines( pl.IMAGE_RATIO_LOW_THRESHOLD   , 0, len_x, color='red'  , alpha=0.8, linestyles='dashed', linewidth=1, label='<30>')    

    plt.ylim(-25, 10000)
    for (x,_,_) in pl.imgAdjustHistory:
        ax.plot(x,-20, 'bo') 
    leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
    leg.set_title('Legend',prop={'size':11})
    ax.set_title('Plot 6 - Curvature - Pxl Thrshld: {:5.1f} OffCt Thrshld: {:3d}'.format(
            pl.IMAGE_RATIO_LOW_THRESHOLD, pl.CURRENT_OFFCTR_ROI_THR ))


