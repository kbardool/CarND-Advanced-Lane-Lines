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
from classes.pipeline import ALFPipeline
from classes.videofile import VideoFile
from classes.camera import Camera
from common.utils import display_one, display_two
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




''' 
#--------------------------------------------------------------------------
#-- MAIN 
#--------------------------------------------------------------------------'''
def main( input_file = 'VIDEO_INPUT', from_frame = 0, to_frame = 999999, output_path = 'output_path', suffix = '', **kwargs): 

    print('--> main routine started at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
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


    ##----------------------------------------------------------
    ## harder challlenge video 
    ##----------------------------------------------------------
    
    Pipeline = ALFPipeline(cameraConfig, **kwargs)
    #                        mode                     =   1, 
    #                        history                  =   5,
    #                        RoI_x_adj                =  30,
    #                        y_src_top                = 520,
    #                        y_src_bot                = 680,
    #                        displayRegionTop         = 520,
    #                        displayRegionBot         = 680,
   
    #                        high_thresholding        = 'cmb_hue_mag_lvl_x',
    #                        normal_thresholding      = 'cmb_hue_mag_lvl_x',
    #                        low_thresholding         = 'cmb_hue_lvl_x',
    #                        vlow_thresholding        = 'cmb_mag_x',
    #                        hisat_thresholding       = 'cmb_mag_x', 
    #                        lowsat_thresholding      = 'cmb_hue_x',
                            
    #                        poly_search_margin       =   48,
    #                        window_search_margin     =   45,
    #                        lane_ratio_threshold     =  2.0,
    #                        lane_ratio_low_threshold =  8.0,
    #                        lane_ratio_high_threshold= 75.0,
    #                        image_ratio_threshold    =   30,
    #                        rse_threshold            =   80,  ## Changed from 120 to 80 3-10-20 
    #                        lane_count_threshold     = 1300,  ## changed from 3000
    #                        off_center_roi_threshold =   50
    # #                      init_window_search_margin =   65,  
    # )
    Pipeline.inVideo  = VideoFile( input_file, mode = 'input' , fromFrame = from_frame, toFrame = to_frame)
    Pipeline.outVideo = VideoFile( input_file, mode = 'output', outputPath = output_path ,suffix = suffix, like = Pipeline.inVideo)

    
    print(' --> ALF Video ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    return Pipeline


def project_overrides(Pipeline):
    
    print('--- project_overrides --------------------------------------')
    ##----------------------------------------------------------
    ## project video - Mode 1
    ##----------------------------------------------------------


    
    Pipeline.itStr = {1: {} , 2: {} }
    for mode in [1,2]:
        for cond in Pipeline.ImageThresholds[mode].keys():
            Pipeline.itStr[mode][cond] = {}
            for thr in Pipeline.ImageThresholds[mode][cond].keys():
                Pipeline.itStr[mode][cond][thr] = str(Pipeline.ImageThresholds[mode][cond][thr]) if Pipeline.ImageThresholds[mode][cond][thr] else 'None'

    print('--- all done' )
    return


def challenge_overrides(Pipeline):

    print('--- challenge_customizations --------------------------------------')
    Pipeline.WINDOW_SRCH_MRGN           =    40   ## default is  40
    Pipeline.POLY_SRCH_MRGN             =    40   ## default is  48
    
    Pipeline.IMAGE_RATIO_HIGH_THRESHOLD =  30.0   ## default is  40
    Pipeline.IMAGE_RATIO_LOW_THRESHOLD  =   0.5   ## default is   2
    Pipeline.LANE_COUNT_THRESHOLD       =   500   ## default is  4500
    Pipeline.LANE_RATIO_HIGH_THRESHOLD  =  74.0   ## default is  60
    Pipeline.LANE_RATIO_LOW_THRESHOLD   =   2.0   ## default is   2
    Pipeline.OFF_CENTER_ROI_THRESHOLD   =    50   ## default is  6-

       
    Pipeline.HIGH_RGB_THRESHOLD         =   205   ## default is  180
    Pipeline.MED_RGB_THRESHOLD          =   170   ## default is  180
    Pipeline.LOW_RGB_THRESHOLD          =   120   ## default is  100
    Pipeline.VLOW_RGB_THRESHOLD         =    90   ## default is   35         
                                              
    Pipeline.XHIGH_SAT_THRESHOLD        =   120   ## default is  120
    Pipeline.HIGH_SAT_THRESHOLD         =    65   ## default is   65
    Pipeline.LOW_SAT_THRESHOLD          =    30   ## default is   20

    # Pipeline.thresholdMethods[1]['xhigh']  =  'cmb_mag_x'             ## HISAT_THRESHOLDING     205 < avg
    # Pipeline.thresholdMethods[1]['high']   =  'cmb_hue_mag_lvl_x'     ## HIGH_THRESHOLDING      170 < avg < 205
    # Pipeline.thresholdMethods[1]['med']    =  'cmb_hue_mag_lvl_x'     ## NORMAL_THRESHOLDING  - 100 < avg < 170
    # Pipeline.thresholdMethods[1]['low']    =  'cmb_hue_mag_x'         ## LOW_THRESHOLDING     -  35 < avg < 130
    # Pipeline.thresholdMethods[1]['vlow']   =  'cmb_hue_mag_x'         ## VLOW_THRESHOLDING    -       avg <  35
    # Pipeline.thresholdMethods[1]['hisat']  =  'cmb_mag_x'             ## HISAT_THRESHOLDING
    # Pipeline.thresholdMethods[1]['lowsat'] =  'cmb_hue_mag_sat'       ## LOWSAT_THRESHOLDING

    Pipeline.thresholdMethods[1]['xhigh']  =  'cmb_rgb_lvl_sat'       ## HISAT_THRESHOLDING     205 < avg
    Pipeline.thresholdMethods[1]['high']   =  'cmb_rgb_lvl_sat'       ## HIGH_THRESHOLDING      170 < avg < 205
    Pipeline.thresholdMethods[1]['med']    =  'cmb_rgb_lvl_sat'       ## NORMAL_THRESHOLDING  - 100 < avg < 170
    Pipeline.thresholdMethods[1]['low']    =  'cmb_rgb_lvl_sat'         ## LOW_THRESHOLDING     -  35 < avg < 130
    Pipeline.thresholdMethods[1]['vlow']   =  'cmb_rgb_lvl_sat_mag'         ## VLOW_THRESHOLDING    -       avg <  35
    Pipeline.thresholdMethods[1]['hisat']  =  'cmb_rgb_lvl_sat'             ## HISAT_THRESHOLDING
    Pipeline.thresholdMethods[1]['lowsat'] =  'cmb_rgb_lvl_sat_mag'         ## LOWSAT_THRESHOLDING
    
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
    #     'x_thr'      : ( 60,255)  ,
    #     'y_thr'      : None  ,
    #     'mag_thr'    : ( 60,255)  ,
    #     'dir_thr'    : ( 40, 65)   ,
    #     'sat_thr'    : None ,
    #     'lvl_thr'    : (252,255) ,  ## (205,255),
    #     'rgb_thr'    : None ,   ## (205,255),
    #     'hue_thr'    : ( 20, 65)
    # }        
    
    Pipeline.ImageThresholds[1]['med']  = {
        'ksize'      : 7         ,
        'x_thr'      : ( 30, 255),
        'y_thr'      : ( 70, 255),
        'mag_thr'    : ( 35, 255),
        'dir_thr'    : ( 40,  65),
        'sat_thr'    : (110, 255),
        'lvl_thr'    : (180, 255),
        'rgb_thr'    : (180, 255),
        'hue_thr'    : ( 20,  65)
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


    Pipeline.itStr = {1: {} , 2: {} }
    for mode in [1,2]:
        for cond in Pipeline.ImageThresholds[mode].keys():
            Pipeline.itStr[mode][cond] = {}
            for thr in Pipeline.ImageThresholds[mode][cond].keys():
                Pipeline.itStr[mode][cond][thr] = str(Pipeline.ImageThresholds[mode][cond][thr]) if Pipeline.ImageThresholds[mode][cond][thr] else 'None'

    print('--- all done' )



def harder_challenge_overrides(Pipeline):
    
    print('--- harder_challenge_overrides --------------------------------------')
    ##----------------------------------------------------------
    ## harder challenge video - Mode 1
    ##----------------------------------------------------------

    Pipeline.IMAGE_RATIO_HIGH_THRESHOLD =  30.0   ## default is  40
    Pipeline.IMAGE_RATIO_LOW_THRESHOLD  =   1.0   ## default is   2
    Pipeline.LANE_COUNT_THRESHOLD       =  4500   ## default is  4500
    Pipeline.LANE_RATIO_HIGH_THRESHOLD  =  74.0   ## default is  60
    Pipeline.LANE_RATIO_LOW_THRESHOLD   =   2.0   ## default is   2
    Pipeline.OFF_CENTER_ROI_THRESHOLD   =   50    ## default is  6-
    
    # Pipeline.WINDOW_SRCH_MRGN         =    45   ## default is  40
    Pipeline.POLY_SRCH_MRGN             =   48    ## default is  45

       
    Pipeline.HIGH_RGB_THRESHOLD         =  205    ## default is  180
    Pipeline.MED_RGB_THRESHOLD          =  170    ## default is  180
    Pipeline.LOW_RGB_THRESHOLD          =  120    ## default is  100
    Pipeline.VLOW_RGB_THRESHOLD         =   90    ## default is   35         

    Pipeline.XHIGH_SAT_THRESHOLD        =  120    ## default is  120
    Pipeline.HIGH_SAT_THRESHOLD         =   65    ## default is   65
    Pipeline.LOW_SAT_THRESHOLD          =   30    ## default is   20

    Pipeline.thresholdMethods[1]['xhigh']  =  'cmb_mag_x'             ## HISAT_THRESHOLDING     205 < avg
    Pipeline.thresholdMethods[1]['high']   =  'cmb_hue_mag_lvl_x'     ## HIGH_THRESHOLDING      170 < avg < 205
    Pipeline.thresholdMethods[1]['med']    =  'cmb_hue_mag_lvl_x'     ## NORMAL_THRESHOLDING  - 100 < avg < 170
    Pipeline.thresholdMethods[1]['low']    =  'cmb_hue_mag_x'         ## LOW_THRESHOLDING     -  35 < avg < 130
    Pipeline.thresholdMethods[1]['vlow']   =  'cmb_hue_mag_x'         ## VLOW_THRESHOLDING    -       avg <  35
    # Pipeline.thresholdMethods[1]['vlow'] =  'cmb_mag_x'             ## VLOW_THRESHOLDING    -       avg <  35
    Pipeline.thresholdMethods[1]['hisat']  =  'cmb_mag_x'             ## HISAT_THRESHOLDING
    Pipeline.thresholdMethods[1]['lowsat'] =  'cmb_hue_mag_sat'       ## LOWSAT_THRESHOLDING


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


    Pipeline.itStr = {1: {} , 2: {} }
    for mode in [1,2]:
        for cond in Pipeline.ImageThresholds[mode].keys():
            Pipeline.itStr[mode][cond] = {}
            for thr in Pipeline.ImageThresholds[mode][cond].keys():
                Pipeline.itStr[mode][cond][thr] = str(Pipeline.ImageThresholds[mode][cond][thr]) if Pipeline.ImageThresholds[mode][cond][thr] else 'None'

    print('--- all done' )
    return




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
