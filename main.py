'''
Created on Apr 12, 2017

@author: kevin.bardool
'''
import sys
# print('classifydotbe : __name__ is ',__name__)
if __name__ == '__main__':
    REL_PATH = './'
    # sys.path.append(REL_PATH)
else:
    REL_PATH   = './'
    
    
import sys, os , io, time , argparse
from datetime                              import datetime
import numpy as np
import cv2, glob, pickle, sys, os , pprint, winsound 
from datetime import datetime, time
from classes.videopipeline import VideoPipeline
from classes.videofile import VideoFile
from classes.camera import Camera
from common.utils import display_one, display_two
from collections import defaultdict

pp = pprint.PrettyPrinter(indent=2, width=100)
print('Current working dir: ', os.getcwd())
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

if '.' not in sys.path:
    print("appending '.' to sys.path")
    sys.path.append('.')

INPUT_PATH            = REL_PATH 
OUTPUT_PATH           = REL_PATH 

print(' main.py module - name is ', __name__)


''' 
#--------------------------------------------------------------------------
#-- MAIN 
#--------------------------------------------------------------------------'''
def main( input_file , from_frame = 0, to_frame = 999999, 
          output_path = 'output_path', 
          config  = None, 
          suffix = datetime.now().strftime("%m-%d-%Y-%H%M_DEV"), 
          **kwargs): 
    assert config in ['project', 'challenge', 'harder_challenge'], "Invalid config parm"
    print('--> main routine started at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    print(' Suffix: ', suffix)
    print(' kwargs: ', kwargs)
    file_sfx = datetime.now().strftime("%m%d%Y@%H%M")

    print('     Input File is   :', input_file)
    print()

    np_format = {}
    np_format['float'] = lambda x: "%12.6f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =100, formatter = np_format)

    with open('./cameraConfig_0.pkl', 'rb') as infile:
        cameraConfig = pickle.load(infile)
        print(' Camera calibration file loaded ...')
    print()
    print(' Camera Calibration Matrix :')
    print(' ---------------------------')
    print(cameraConfig.cameraMatrix)


    Pipeline = VideoPipeline(cameraConfig, **kwargs)

    Pipeline.inVideo  = VideoFile( input_file, mode = 'input' , fromFrame = from_frame, toFrame = to_frame)
    Pipeline.outVideo = VideoFile( input_file, mode = 'output', outputPath = output_path ,suffix = suffix, like = Pipeline.inVideo)

    if config == 'project':
        project_overrides(Pipeline)
    elif config == 'challenge':
        challenge_overrides(Pipeline)
    elif config == 'harder_challenge':
        harder_challenge_overrides(Pipeline)

    Pipeline.thresholds_to_str()

    print('--- Video Pipeline setup complete:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    return Pipeline


def project_overrides(Pipeline):
    
    print('--- project_overrides --------------------------------------')

    # Pipeline.HIGH_RGB_THRESHOLD         =  180    ## default is  180
    # Pipeline.MED_RGB_THRESHOLD          =  180    ## default is  180
    # Pipeline.LOW_RGB_THRESHOLD          =  100    ## default is  100
    # Pipeline.VLOW_RGB_THRESHOLD         =   35    ## default is   35         

    # Pipeline.XHIGH_SAT_THRESHOLD        =  120    ## default is  120
    # Pipeline.HIGH_SAT_THRESHOLD         =   65    ## default is   65
    # Pipeline.LOW_SAT_THRESHOLD          =   30    ## default is   20

    # Pipeline.thresholdMethods[1]['xhigh']  =  'cmb_mag_x'         ## HISAT_THRESHOLDING     205 < avg
    # Pipeline.thresholdMethods[1]['high']   =  'cmb_mag_x'         ## HIGH_THRESHOLDING      170 < avg < 205
    # Pipeline.thresholdMethods[1]['med']    =  'cmb_rgb_lvl_sat'   ## NORMAL_THRESHOLDING  - 100 < avg < 170
    # Pipeline.thresholdMethods[1]['low']    =  'cmb_mag_xy'        ## LOW_THRESHOLDING     -  35 < avg < 130
    # Pipeline.thresholdMethods[1]['vlow']   =  'cmb_mag_xy'        ## VLOW_THRESHOLDING    -       avg <  35
    # Pipeline.thresholdMethods[1]['hisat']  =  'cmb_mag_x'         ## HISAT_THRESHOLDING
    # Pipeline.thresholdMethods[1]['lowsat'] =  'cmb_hue_x'         ## LOWSAT_THRESHOLDING
    
    print('--- all done' )
    return


def challenge_overrides(Pipeline):
    
    print('--- challenge_customizations --------------------------------------')
    # Pipeline.WINDOW_SRCH_MRGN           =    40   ## default is  40
    # Pipeline.POLY_SRCH_MRGN             =    40   ## default is  48
    
    # Pipeline.IMAGE_RATIO_HIGH_THRESHOLD =  30.0   ## default is  40
    # Pipeline.IMAGE_RATIO_LOW_THRESHOLD  =   0.5   ## default is   2

    # Pipeline.LANE_COUNT_THRESHOLD       =   500   ## default is  4500
    # Pipeline.LANE_RATIO_HIGH_THRESHOLD  =  74.0   ## default is  60
    # Pipeline.LANE_RATIO_LOW_THRESHOLD   =   2.0   ## default is   2

    # Pipeline.OFF_CENTER_ROI_THRESHOLD   =    50   ## default is  6-
       
    # Pipeline.HIGH_RGB_THRESHOLD         =   205   ## default is  180
    # Pipeline.MED_RGB_THRESHOLD          =   170   ## default is  180
    # Pipeline.LOW_RGB_THRESHOLD          =   120   ## default is  100
    # Pipeline.VLOW_RGB_THRESHOLD         =    90   ## default is   35         
                                              
    # Pipeline.XHIGH_SAT_THRESHOLD        =   120   ## default is  120
    # Pipeline.HIGH_SAT_THRESHOLD         =    65   ## default is   65
    # Pipeline.LOW_SAT_THRESHOLD          =    30   ## default is   20

    # Pipeline.thresholdMethods[1]['xhigh']  =  'cmb_rgb_lvl_sat'       ## HISAT_THRESHOLDING     205 < avg
    # Pipeline.thresholdMethods[1]['high']   =  'cmb_rgb_lvl_sat'       ## HIGH_THRESHOLDING      170 < avg < 205
    # Pipeline.thresholdMethods[1]['med']    =  'cmb_rgb_lvl_sat'       ## NORMAL_THRESHOLDING  - 100 < avg < 170
    # Pipeline.thresholdMethods[1]['low']    =  'cmb_rgb_lvl_sat'         ## LOW_THRESHOLDING     -  35 < avg < 130
    # Pipeline.thresholdMethods[1]['vlow']   =  'cmb_rgb_lvl_sat_mag'         ## VLOW_THRESHOLDING    -       avg <  35
    
    # # Pipeline.thresholdMethods[1]['hisat']  =  'cmb_rgb_lvl_sat'             ## HISAT_THRESHOLDING
    # Pipeline.thresholdMethods[1]['hisat']  =  'cmb_mag_x'             ## HISAT_THRESHOLDING
    # Pipeline.thresholdMethods[1]['lowsat'] =  'cmb_rgb_lvl_sat_mag'         ## LOWSAT_THRESHOLDING
    
    ##----------------------------------------------------------
    ## challlenge video - Mode 1
    ##----------------------------------------------------------

    # Pipeline.ImageThresholds[1]['normal'] = {
    #     'ksize'      : 7         , 
    #     'x_thr'      : (30,255)  ,
    #     'y_thr'      : (70,255)  ,
    #     'mag_thr'    : (10,50)   ,
    #     'dir_thr'    : (0,30)    ,
    #     'sat_thr'    : (60, 255) ,   ### (80, 255) ,
    #     'lvl_thr'    : (180,255) ,
    #     'rgb_thr'    : (180,255) 
    # }

    # Pipeline.ImageThresholds[1]['dark'] = {
    #     'ksize'      : 7         ,
    #     'x_thr'      : (30,255)  ,
    #     'y_thr'      : (45,255)  ,
    #     'mag_thr'    : (55,255)  ,
    #     'dir_thr'    : (40,65)   ,
    #     'sat_thr'    : (160,255) ,
    #     'lvl_thr'    : (180,255),  ## (205,255),
    #     'rgb_thr'    : (180,255)   ## (205,255),
    # }        

    Pipeline.ImageThresholds[1]['xhigh']  = {
        'ksize'      : 7         ,
        'x_thr'      : ( 60,255) ,
        'y_thr'      : None  ,
        'mag_thr'    : ( 60,255) ,
        'dir_thr'    : ( 40, 65) ,
        'sat_thr'    : None ,
        'lvl_thr'    : (252, 255) ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : ( 25, 50)
    }   
    Pipeline.ImageThresholds[1]['high']  = {
        'ksize'      : 7         ,
        'x_thr'      : ( 60,255)  ,
        'y_thr'      : None  ,
        'mag_thr'    : ( 60,255)  ,
        'dir_thr'    : ( 40, 65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : (252,255) ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : ( 25, 65)
    }        

    ## Image Thresholding params challenge video 1/28/20------
    # ksize      = 7
    # grad_x_thr = ( 30,255)
    # grad_y_thr = ( 70,255)
    # mag_thr    = ( 35,255)
    # dir_thr    = ( 40, 65)
    # sat_thr    = (110,255)
    # lvl_thr    = (205,255)
    # rgb_thr    = (205,255)
    ##--------------------------------------------------------

    # Pipeline.ImageThresholds[1]['med']  = {
    #     'ksize'      : 7         ,
    #     'x_thr'      : ( 30, 255),
    #     'y_thr'      : ( 70, 255),
    #     'mag_thr'    : ( 35, 255),
    #     'dir_thr'    : ( 40,  65),
    #     'sat_thr'    : (110, 255),
    #     'lvl_thr'    : (180, 255),
    #     'rgb_thr'    : (180, 255),
    #     'hue_thr'    : ( 20,  65)
    # }        
    # Pipeline.ImageThresholds[1]['med']  = {
    #     'ksize'      : 7         ,
    #     'x_thr'      : ( 60,255)  ,
    #     'y_thr'      : None  ,
    #     'mag_thr'    : ( 60,255)  ,
    #     'dir_thr'    : ( 40, 65)   ,
    #     'sat_thr'    : None ,
    #     'lvl_thr'    : (252,255) ,  ## (205,255),
    #     'rgb_thr'    : None ,   ## (205,255),
    #     'hue_thr'    : ( 20, 65)
    # }        
    
    ## modified June 25 2020
    # Pipeline.ImageThresholds[1]['med']  = {
    #     'ksize'      : 7         ,
    #     'x_thr'      : ( 15, 30),
    #     'y_thr'      : None,
    #     'mag_thr'    : ( 10,  50),
    #     'dir_thr'    : ( 45,  65),
    #     'sat_thr'    : ( 90, 175),
    #     'lvl_thr'    : (200, 255),
    #     'rgb_thr'    : (200, 255),
    #     'hue_thr'    : (  5,  35)
    # }        

    ## modified September 17 2020
    Pipeline.ImageThresholds[1]['med']  = {
        'ksize'      : 7,
        'x_thr'      : ( 15, 30),
        'y_thr'      : None,
        'mag_thr'    : ( 10, 50),
        'dir_thr'    : ( 45, 65),
        'rgb_thr'    : (170,255),
        'hue_thr'    : (  5, 35),
        'lvl_thr'    : (170,255),
        'sat_thr'    : ( 90,175),
    }
    
    Pipeline.ImageThresholds[1]['low']  = {
        'ksize'      : 7         ,
        'x_thr'      : (65,255)  ,
        'y_thr'      : None      ,
        'mag_thr'    : (55,255)  ,
        'dir_thr'    : (40,65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : (250, 255) ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : None  ## 0,25  
    }        

    Pipeline.ImageThresholds[1]['vlow']  = {
        'ksize'      : 7         ,
        'x_thr'      : (35,255)  ,
        'y_thr'      : None      ,
        'mag_thr'    : (35,255)  ,
        'dir_thr'    : (40,65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : None ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : (20,35)    
    }

    Pipeline.ImageThresholds[1]['lowsat']= {
        'ksize'      : 7         ,
        'x_thr'      : ( 55,255)  ,
        'y_thr'      : None      ,
        'mag_thr'    : ( 45,255)      ,   ### (25,250)  ,
        'dir_thr'    : ( 40, 65)      ,
        'sat_thr'    : None      ,
        'lvl_thr'    : (140,255) ,
        'rgb_thr'    : None      ,
        'hue_thr'    : ( 20, 35)
    }
    Pipeline.ImageThresholds[1]['hisat']  = {
        'ksize'      : 7         ,
        'x_thr'      : (50,255)  ,
        'y_thr'      : None      ,
        'mag_thr'    : (50,255)  ,
        'dir_thr'    : (40,65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : None ,   ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : (25,35)
    }

    Pipeline.thresholds_to_str()

    print('--- all done' )



def harder_challenge_overrides(Pipeline):
    
    print('--- harder_challenge_overrides --------------------------------------')
    ##----------------------------------------------------------
    ## harder challenge video - Mode 1
    ##----------------------------------------------------------

    # Pipeline.IMAGE_RATIO_HIGH_THRESHOLD =  30.0   ## default is  40
    # Pipeline.IMAGE_RATIO_LOW_THRESHOLD  =   1.0   ## default is   2

    # Pipeline.LANE_COUNT_THRESHOLD       =  4500   ## default is  4500
    # Pipeline.LANE_RATIO_HIGH_THRESHOLD  =  74.0   ## default is  60
    # Pipeline.LANE_RATIO_LOW_THRESHOLD   =   2.0   ## default is   2

    # Pipeline.OFF_CENTER_ROI_THRESHOLD   =   50    ## default is  6-
    
    # Pipeline.HIGH_RGB_THRESHOLD         =  205    ## default is  180
    # Pipeline.MED_RGB_THRESHOLD          =  170    ## default is  180
    # Pipeline.LOW_RGB_THRESHOLD          =  120    ## default is  100
    # Pipeline.VLOW_RGB_THRESHOLD         =   90    ## default is   35         

    # Pipeline.XHIGH_SAT_THRESHOLD        =  120    ## default is  120
    # Pipeline.HIGH_SAT_THRESHOLD         =   65    ## default is   65
    # Pipeline.LOW_SAT_THRESHOLD          =   30    ## default is   20

 

    # Pipeline.ImageThresholds[1]['xhigh']  = {
    #     'ksize'      : 7         ,
    #     'x_thr'      : (45,255)  ,
    #     'y_thr'      : None      ,
    #     'mag_thr'    : (35,255)  ,
    #     'dir_thr'    : (40,65)   ,
    #     'sat_thr'    : None ,
    #     'lvl_thr'    : None ,  ## (205,255),
    #     'rgb_thr'    : None ,   ## (205,255),
    #     'hue_thr'    : ( 15, 25)
    # }
    # Pipeline.ImageThresholds[1]['high']  = {
    #     'ksize'      : 7         ,
    #     'x_thr'      : ( 45,255)  ,
    #     'y_thr'      : None  ,
    #     'mag_thr'    : ( 45,255)  ,
    #     'dir_thr'    : ( 40, 65)   ,
    #     'sat_thr'    : None ,
    #     'lvl_thr'    : (250,255) ,  ## (205,255),
    #     'rgb_thr'    : (250,255),   ## (205,255),
    #     'hue_thr'    : ( 20, 65)
    # }        
    # Pipeline.ImageThresholds[1]['lowsat']= {
    #     'ksize'      : 7         ,
    #     'x_thr'      : ( 65,255)  ,
    #     'y_thr'      : None      ,
    #     'mag_thr'    : ( 55,255)      ,   ### (25,250)  ,
    #     'dir_thr'    : ( 40, 65)      ,
    #     'sat_thr'    : None      ,
    #     'lvl_thr'    : None      ,
    #     'rgb_thr'    : None      ,
    #     'hue_thr'    : ( 20, 35)
    # }
    # Pipeline.ImageThresholds[1]['lowsat']= {
    #     'ksize'      : 7         ,
    #     'x_thr'      : ( 65,255)  ,
    #     'y_thr'      : None      ,
    #     'mag_thr'    : ( 55,255)      ,   ### (25,250)  ,
    #     'dir_thr'    : ( 40, 65)      ,
    #     'sat_thr'    : ( 50,255)  ,
    #     'lvl_thr'    : (150,255) ,
    #     'rgb_thr'    : None      ,
    #     'hue_thr'    : ( 10, 35)
    # }

    Pipeline.ImageThresholds[1]['xhigh']  = {
        'ksize'      : 7         ,
        'x_thr'      : ( 60,255) ,
        'y_thr'      : None  ,
        'mag_thr'    : ( 60,255) ,
        'dir_thr'    : ( 40, 65) ,
        'sat_thr'    : None ,
        'lvl_thr'    : (252, 255) ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : ( 25, 50)
    }   
    Pipeline.ImageThresholds[1]['high']  = {
        'ksize'      : 7         ,
        'x_thr'      : ( 60,255)  ,
        'y_thr'      : None  ,
        'mag_thr'    : ( 60,255)  ,
        'dir_thr'    : ( 40, 65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : (252,255) ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : ( 25, 65)
    }      


    # Pipeline.ImageThresholds[1]['med']  = {
#     'ksize'      : 7         ,
#     'x_thr'      : (45,255)  ,
#     'y_thr'      : None  ,
#     'mag_thr'    : (45,255)  ,
#     'dir_thr'    : (40,65)   ,
#     'sat_thr'    : None ,
#     'lvl_thr'    : (250,255) ,  ## (205,255),
#     'rgb_thr'    : None ,   ## (205,255),
#     'hue_thr'    : ( 20, 65)
# }        
  
    Pipeline.ImageThresholds[1]['med']  = {
        'ksize'      : 7         ,
        'x_thr'      : ( 60,255)  ,
        'y_thr'      : None  ,
        'mag_thr'    : ( 60,255)  ,
        'dir_thr'    : ( 40, 65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : (252,255) ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : ( 20, 65)
    }        

    Pipeline.ImageThresholds[1]['low']  = {
        'ksize'      : 7         ,
        'x_thr'      : (65,255)  ,
        'y_thr'      : None      ,
        'mag_thr'    : (55,255)  ,
        'dir_thr'    : (40,65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : (250, 255) ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : None  ## 0,25  
    }        

    Pipeline.ImageThresholds[1]['vlow']  = {
        'ksize'      : 7         ,
        'x_thr'      : (35,255)  ,
        'y_thr'      : None      ,
        'mag_thr'    : (35,255)  ,
        'dir_thr'    : (40,65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : None ,  ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : (20,35)    
    }

    Pipeline.ImageThresholds[1]['lowsat']= {
        'ksize'      : 7         ,
        'x_thr'      : ( 55,255)  ,
        'y_thr'      : None      ,
        'mag_thr'    : ( 45,255)      ,   ### (25,250)  ,
        'dir_thr'    : ( 40, 65)      ,
        'sat_thr'    : None      ,
        'lvl_thr'    : (140,255) ,
        'rgb_thr'    : None      ,
        'hue_thr'    : ( 20, 35)
    }
    Pipeline.ImageThresholds[1]['hisat']  = {
        'ksize'      : 7         ,
        'x_thr'      : (50,255)  ,
        'y_thr'      : None      ,
        'mag_thr'    : (50,255)  ,
        'dir_thr'    : (40,65)   ,
        'sat_thr'    : None ,
        'lvl_thr'    : None ,   ## (205,255),
        'rgb_thr'    : None ,   ## (205,255),
        'hue_thr'    : (25,35)
    }


    Pipeline.thresholds_to_str(debug = True)

    print('--- all done' )
    return

def get_pipeline_parms(config = None):
    assert config in ['project', 'challenge', 'harder_challenge'], "Invalid input_file"
    pipeline_parms = defaultdict(dict)

    ## specific settings for project video -----------
    pipeline_parms['project'] = {
        'mode'                     :     1,
        'history'                  :     8,

        'y_src_top'                :   480,
        'y_src_bot'                :   690,
        'RoI_x_adj'                :    30,
        'displayRegionTop'         :   480,
        'displayRegionBot'         :   690,
        'lowlight_threshold'       :   'cmb_rgb_lvl_sat',

        'rgb_mean_threshold'       :   175,
        'init_window_search_margin':    65,
        'window_search_margin'     :    65,
        'poly_search_margin'       :    50,
        'pixel_ratio_threshold'    :    30,
        'offcntr_roi_threshold'    :    80,

        'high_rgb_threshold'       :   180,
        'med_rgb_threshold'        :   180,
        'low_rgb_threshold'        :   100,
        'vlow_rgb_threshold'       :    35,
 
        'xhigh_sat_threshold'      :   120,
        'high_sat_threshold'       :    65,
        'low_sat_threshold'        :    30, 

        'xhigh_thresholding'       :  'cmb_mag_x' ,        ## HISAT_THRESHOLDING     205 < avg      
        'high_thresholding'        :  'cmb_mag_x' ,        ## HIGH_THRESHOLDING      170 < avg < 205
        'med_thresholding'         :  'cmb_rgb_lvl_sat',   ## NORMAL_THRESHOLDING  - 100 < avg < 170
        'low_thresholding'         :  'cmb_mag_xy',        ## LOW_THRESHOLDING     -  35 < avg < 130
        'vlow_thresholding'        :  'cmb_mag_xy',        ## VLOW_THRESHOLDING    -       avg <  35

        'hisat_thresholding'       :  'cmb_mag_x' ,        ## HISAT_THRESHOLDING                    
        'lowsat_thresholding'      :  'cmb_hue_x'         ## LOWSAT_THRESHOLDING                   
    }



    pipeline_parms['challenge'] = {
        'mode'                     :     1,
        'min_poly_degree'          :     1,
        'min_x_spread'             :    50,
        'min_y_spread'             :   250,
             
        'history'                  :     8,
        'compute_history'          :     8,     ## '2' for harder_challenge_video
             
        'window_search_margin'     :    40,  ## 40,     ## default   65 <- 55
        'poly_search_margin'       :    40,  ## 40,     ## default   50 <- 45 

        'image_ratio_high_threshold': 30.0,
        'image_ratio_low_threshold' :  0.5,

        'lane_count_threshold'     :   500,
        'lane_ratio_high_threshold':  74.0,
        'lane_ratio_low_threshold' :   2.0,

        
        'off_center_roi_threshold' :   50,  ## default   60 - original  80 

        'high_rgb_threshold'       :   205,
        'med_rgb_threshold'        :   170,
        'low_rgb_threshold'        :   120,
        'vlow_rgb_threshold'       :    90,
 
        'xhigh_sat_threshold'      :   120,
        'high_sat_threshold'       :    65,
        'low_sat_threshold'        :    30, 

        'xhigh_thresholding'       :  'cmb_rgb_lvl_sat',
        'high_thresholding'        :  'cmb_rgb_lvl_sat',
        'med_thresholding'         :  'cmb_rgb_lvl_sat',
        'low_thresholding'         :  'cmb_rgb_lvl_sat',
        'vlow_thresholding'        :  'cmb_rgb_lvl_sat_mag',
 
        'hisat_thresholding'       :  'cmb_mag_x', 
        'lowsat_thresholding'      :  'cmb_rgb_lvl_sat_mag',        
             
        'y_src_top'                :  480,     ## default  480
        'y_src_bot'                :  700,     ## default  720
        'RoI_x_adj'                :   30,     ## Default   25
 
        'displayRegionTop'         :  500,     ## default  480
        'displayRegionBot'         :  700,     ## default  720
    } 
    
    
    pipeline_parms['harder_challenge'] = {
        'mode'                     :   1, 

        'history'                  :   5,

        'image_ratio_high_threshold': 30.0,
        'image_ratio_low_threshold' :  1.0,

        'lane_count_threshold'     :  1300,   ## not sure if its 1300 or 4500 (10-7-2020),
        'lane_ratio_high_threshold':  74.0,
        'lane_ratio_low_threshold' :   8.0,            

        'rse_threshold'            :   80,  ## Changed from 120 to 80 3-10-20 
        
        'off_center_roi_threshold' :   50,

        'high_rgb_threshold'       :   205,
        'med_rgb_threshold'        :   170,
        'low_rgb_threshold'        :   120,
        'vlow_rgb_threshold'       :    90,
 
        'xhigh_sat_threshold'      :   120,
        'high_sat_threshold'       :    65,
        'low_sat_threshold'        :    30, 

        'xhigh_thresholding'       : 'cmb_mag_x',
        'high_thresholding'        : 'cmb_hue_mag_lvl_x',
        'med_thresholding'         : 'cmb_hue_mag_lvl_x',
        'low_thresholding'         : 'cmb_hue_mag_x',
        'vlow_thresholding'        : 'cmb_hue_mag_x',
            
        'hisat_thresholding'       : 'cmb_mag_x', 
        'lowsat_thresholding'      : 'cmb_hue_mag_sat',

        'poly_search_margin'       :   48,
        'window_search_margin'     :   45,
        

        'y_src_top'                : 520,
        'y_src_bot'                : 680,
        'RoI_x_adj'                :  30,

        'displayRegionTop'         : 520,
        'displayRegionBot'         : 680
    }

    # pipeline_parms['challenge'] = {
    #     'mode':                       1,
    #     'history':                    8,
    #     'compute_history':            8,     ## '2' for harder_challenge_video
    #     'min_poly_degree':            1,
    #     'RoI_x_adj':                 30,     ## Default   25
            
    #     'y_src_top':                480,     ## default  480
    #     'y_src_bot':                700,     ## default  720
    #     'displayRegionTop':         500,     ## default  480
    #     'displayRegionBot':         700,     ## default  720
            
    #     'min_x_spread':              50,
    #     'min_y_spread':             250,
            
    #     'high_rgb_threshold'       : 205,
    #     'med_rgb_threshold'        : 170,
    #     'low_rgb_threshold'        : 120,
    #     'vlow_rgb_threshold'       :  90,

    #     'xhigh_sat_threshold'      : 125,
    #     'high_sat_threshold'       :  65,
    #     'low_sat_threshold'        :  30 , 

    #     'xhigh_thresholding'       : 'cmb_rgb_lvl_sat',
    #     'high_thresholding'        : 'cmb_rgb_lvl_sat',
    #     'normal_thresholding'      : 'cmb_rgb_lvl_sat',
    #     'low_thresholding'         : 'cmb_rgb_lvl_sat',
    #     'vlow_thresholding'        : 'cmb_rgb_lvl_sat_mag',

    #     'hisat_thresholding'       : 'cmb_mag_x', 
    #     'lowsat_thresholding'      : 'cmb_rgb_lvl_sat_mag',    
            
    #     'window_search_margin'     :  48,  ## 40,     ## default   65 <- 55
    #     'poly_search_margin'       :  45,  ## 40,     ## default   50 <- 45 
    #     'off_center_roi_threshold' :  50,  ## default   60 - original  80 
            
    #     'lane_ratio_threshold'     :  2.0,
    #     'lane_ratio_low_threshold' :  8.0,
    #     'lane_ratio_high_threshold': 75.0,
    #     'image_ratio_threshold'    :   30,
    #     'rse_threshold'            :   80, ## Changed from 120 to 80 3-10-20 
    #     'lane_count_threshold'     : 1300  ## changed from 3000    
        
    # # 'init_window_search_margin' =   65,  
    
    # }
    # pipeline_parms['harder_challenge'] = {
        # 'mode'                     :   1, 
        # 'history'                  :   5,
        # 'RoI_x_adj'                :  30,
            
        # 'y_src_top'                : 520,
        # 'y_src_bot'                : 680,
        # 'displayRegionTop'         : 520,
        # 'displayRegionBot'         : 680,

        # 'xhigh_thresholding'       : 'cmb_mag_x',
        # 'high_thresholding'        : 'cmb_hue_mag_lvl_x',
        # 'normal_thresholding'      : 'cmb_hue_mag_lvl_x',
        # 'low_thresholding'         : 'cmb_hue_mag_x',
        # 'vlow_thresholding'        : 'cmb_hue_mag_x',
            
        # 'hisat_thresholding'       : 'cmb_mag_x', 
        # 'lowsat_thresholding'      : 'cmb_hue_mag_sat',

        # 'poly_search_margin'       :   48,
        # 'window_search_margin'     :   45,
        # 'lane_ratio_threshold'     :  2.0,
        # 'lane_ratio_low_threshold' :  8.0,
        # 'lane_ratio_high_threshold': 75.0,
        # 'image_ratio_threshold'    :   30,
        # 'rse_threshold'            :   80,  ## Changed from 120 to 80 3-10-20 
        # 'lane_count_threshold'     : 1300,  ## changed from 3000
        # 'off_center_roi_threshold' :   50
    # }

    return pipeline_parms[config]



#--------------------------------------------------------------------------
#  Main Driver 
#--------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Models   = {     'HtmlText': ['LR_05_13_2017@1144'] ,
                    #  'MetaData': ['RF_05_12_2017@1731'],
                    #  'MetaText': ['LSVC_05_12_2017@1724']}
    """
    Verify input parms - Get input/output filenames from command line
    """
    parser = argparse.ArgumentParser(description='Load csv file of domains to classify ')
    
    parser.add_argument('input_file', metavar='Input_filename', default= 'input_file',
                        help='Input file: Must be in .mp4 format ')

    parser.add_argument('output_file', metavar='Output_filename', default= 'input_file',
                        help='Output destination ' )    

    # parser.add_argument('-fs', metavar='Feature set',  choices=['htmltext', 'metatext', 'metadata', 'all'], 
                        # default= 'htmltext', dest='classify_fs',
                        # help='Feature set to classify can be one of the following: htmltext, metatext, or metadata')    
    # parser.add_argument('-cs', metavar='Crawl sites',  choices=['y', 'n'], 
                        # default= 'y', dest='crawl_sites',
                        # help='Feature set to classify can be one of the following: htmltext, metatext, or metadata')                         
    
    args = parser.parse_args()
    start = time.time()
    print(' input file: ',args.input_file, '  output__file: ', args.output_file)
    
    # main(Models, args.input_file, args.output_file, args.crawl_sites, args.classify_fs)
    print()
    elapsed_time = time.time() - start
    print('     Elapsed time: %.2f seconds'%(elapsed_time))

    exit(' ALF Video completed normally')
