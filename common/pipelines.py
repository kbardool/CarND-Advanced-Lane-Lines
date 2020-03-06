import sys, os, pprint, pickle
import numpy as np
import matplotlib.image as mpimg
from common.utils   import *
from common.sobel   import apply_thresholds_v1
from classes.plotting import PlotDisplay
pp = pprint.PrettyPrinter(indent=2, width=100)


def imagePipeline(filename = None, camera = None, mode = 0 , **kwargs):
    assert filename is not None, ' Filename must be specified' 
    assert camera   is not None, ' Camera object must be specified'
    assert 0 < mode < 3        , ' mode must be 1 or 2'

    # pp.pprint(kwargs)
    # print('-'*30)    
    print('\n Pipeline Input Parms : ')
    print('-'*30)
    img_filename_ext = os.path.basename(filename)
    img_filename = os.path.splitext(img_filename_ext)
    print(' Input image: ', img_filename_ext)

    image = mpimg.imread(filename)
    height          = image.shape[0]
    width           = image.shape[1]
    camera_x        = width //2
    camera_y        = height
    
    print(' height:  {}    width:  {}    camera_x:  {}    camera_y: {}'.format(height, width, camera_x, camera_y))
    
    # mode            = kwargs.get('mode'   , 1)
    debug           = kwargs.get('debug'  , False)
    debug2          = kwargs.get('debug2' , False)
    debug3          = kwargs.get('debug3' , False)
    displayResults  = kwargs.get('displayResults' , False)
    frameTitle      = kwargs.get('frameTitle'     , '')
    thresholdKey    = kwargs.get('thresholdKey'   , 'cmb_rgb_lvl_sat_mag_x')
    nwindows        = kwargs.get('nwindows'       ,  7)    
    window_margin   = kwargs.get('window_margin'  , 40)    

    ksize           = kwargs.get('ksize'          ,  7)
    x_thr           = kwargs.get('x_thr'          , (30,110))
    y_thr           = kwargs.get('y_thr'          , (30,110))
    dir_thr         = kwargs.get('dir_thr'        , (40,65))
    mag_thr         = kwargs.get('mag_thr'        , (65,255))
    # sat_thr         = kwargs.get('sat_thr'        , (130,255))
    sat_thr         = kwargs.get('sat_thr'        , (200,255))

    x_thr2          = x_thr
    y_thr2          = None
    mag_thr2        = (50,255) 
    dir_thr2        = (0,10)
    sat_thr2        = (80, 255)     
    ## Source/Dest points for Perspective Transform   
    x_src_top_left  = kwargs.get('x_src_top_left' ,   570)  ## 580 -> 573
    x_src_top_right = kwargs.get('x_src_top_right',   714)
    x_src_bot_left  = kwargs.get('x_src_bot_left' ,   220) 
    x_src_bot_right = kwargs.get('x_src_bot_right',  1090) 
    y_src_bot       = kwargs.get('y_src_bot'      ,   700)  ## image.shape[0] - 20
    y_src_top       = kwargs.get('y_src_top'      ,   465)  ## 460 -> 465 y_src_bot - 255

    x_dst_left      = kwargs.get('x_dst_left'     ,   300)
    x_dst_right     = kwargs.get('x_dst_right'    ,  1000)
    y_dst_top       = kwargs.get('y_dst_top'      ,     0)
    y_dst_bot       = kwargs.get('y_dst_bot'      ,  height - 1)    

    ## Source/Dest points for Perspective Transform  (FROM VIDEO PIPELINE)      
    # x_src_top_left  = kwargs.get('x_src_top_left' ,  600)
    # x_src_top_right = kwargs.get('x_src_top_right',  740)
    # x_src_bot_left  = kwargs.get('x_src_bot_left' ,  295)
    # x_src_bot_right = kwargs.get('x_src_bot_right', 1105)
    # y_src_top       = kwargs.get('y_src_bot'      ,  480)
    # y_src_bot       = kwargs.get('y_src_top'      ,  700)  

    # x_dst_left      = kwargs.get('x_dst_left'     ,   300)
    # x_dst_right     = kwargs.get('x_dst_right'    ,  1000)
    # y_dst_top       = kwargs.get('y_dst_top'      ,     0)
    # y_dst_bot       = kwargs.get('y_dst_bot'      , height - 1)    
    
    
    
    src_points_list = [(x_src_top_left , y_src_top),
                       (x_src_top_right, y_src_top),
                       (x_src_bot_right, y_src_bot),      
                       (x_src_bot_left , y_src_bot)]
                            
    src_points = np.array( src_points_list, dtype = np.float32)  

    dst_points_list = [ (x_dst_left , y_dst_top), 
                        (x_dst_right, y_dst_top), 
                        (x_dst_right, y_dst_bot), 
                        (x_dst_left , y_dst_bot)]
                            
    dst_points = np.array( dst_points_list, dtype = np.float32)


    ##----------------------------------------------------------------------------
    # ksize = 19 # Choose a larger odd number to smooth gradient measurements
    # grad_x_thr = (70,100)
    # grad_y_thr = (80,155)
    # mag_thr    = (90,160)    
    # theta1     = 45
    # theta2     = 67
    # sat_thr    = (90,255)
    ##----------------------------------------------------------------------------

    ##----------------------------------------------------------------------------
    # RoI_x_top_left  = x_src_top_left  ## - 3
    # RoI_x_top_right = x_src_top_right ## + 3
    # RoI_x_bot_left  = x_src_bot_left  ## - 3
    # RoI_x_bot_right = x_src_bot_right ## + 3
    # RoI_y_bot       = y_src_bot       ## + 3
    # RoI_y_top       = y_src_top       ## - 3
    #
    # print(' Y bottom: ', RoI_y_bot, '   y_top : ', RoI_y_top)
    # RoI_vertices_list = [(RoI_x_bot_left , RoI_y_bot), 
    #                      (RoI_x_top_left , RoI_y_top), 
    #                      (RoI_x_top_right, RoI_y_top), 
    #                      (RoI_x_bot_right, RoI_y_bot)]
    # RoI_vertices      = np.array([RoI_vertices_list],dtype = np.int32)
    ##----------------------------------------------------------------------------
    # warpedRoIVertices_list = [(x_transform_left-2 , 0), 
    #                           (x_transform_right+2, 0), 
    #                           (x_transform_right+2, height-1), 
    #                           (x_transform_left-2 , height-1)]
    # warpedRoIVertices = np.array([warpedRoIVertices_list],dtype = np.int32)
    # src_points = np.array([ ( 611, RoI_y_top), 
    #                         ( 666, RoI_y_top), 
    #                         (1055, RoI_y_bot), 
    #                          (250, RoI_y_bot)],dtype = np.float32)
    # src_points = np.array([ ( 568, RoI_y_top), 
    #                         ( 723, RoI_y_top), 
    #                         (1090, RoI_y_bot), 
    #                         ( 215, RoI_y_bot)],dtype = np.float32)
    # dst_points = np.array([ (x_transform_left,    2), 
    #                         (x_transform_right,   2), 
    #                         (x_transform_right, 718), 
    #                         (x_transform_left,  718)],dtype = np.float32)
    # src_points = np.array([ ( 595, RoI_y_top),
    #                         ( 690, RoI_y_top),
    #                         (1087, RoI_y_bot),         ### was 692
    #                         ( 228, RoI_y_bot)],dtype = np.float32)   ### was 705
    ##----------------------------------------------------------------------------
    
    ##----------------------------------------------------------------------------
    ## Perspective Transform Source/Dest points
    #
    # straight_line1_transform_points = {
    # 'x_src_top_left'  :  575,
    # 'x_src_top_right' :  708,
    #
    # 'x_src_bot_left'  :  220,  ## OR 230 
    # 'x_src_bot_right' : 1090,  ## OR 1100
    #
    # 'y_src_bot'       :  700 , ## image.shape[0] - 20
    # 'y_src_top'       :  460 , ## y_src_bot - 255
    #
    # 'x_dst_left'      : 300,
    # 'x_dst_right'     : 1000,
    # 'y_dst_top'       : 0,
    # 'y_dst_bot'       : height - 1
    # }
    #
    ##----------------------------------------------------------------------------



    ###----------------------------------------------------------------------------------------------
    ###  Remove camera distortion and apply perspective transformation
    ###----------------------------------------------------------------------------------------------
    imgUndist = camera.undistortImage(image)

    imgWarped, M, Minv = perspectiveTransform(imgUndist, src_points, dst_points, debug = False)


                
                
    ##----------------------------------------------------------------------------
    ## Image Tresholding
    ##----------------------------------------------------------------------------
    imgThrshldDict = apply_thresholds_v1(imgUndist, ksize=ksize, 
                                         x_thr = x_thr, y_thr = y_thr, 
                                       mag_thr = mag_thr, dir_thr = dir_thr , 
                                       sat_thr = sat_thr, debug = debug2)
    
    imgThrshld = imgThrshldDict[thresholdKey]  
    imgThrshldWarped, M, Minv = perspectiveTransform(imgThrshld, src_points, dst_points, debug = False)
    # for i,v in imgThrshldDict.items():
        # print(' key: {:30s}    shape: {}    min: {}   max: {} '.format(i, v.shape, v.min(), v.max()))


    ###----------------------------------------------------------------------------------------------
    ### Display undistorted color image & perpective transformed image -- With RoI line display
    ###----------------------------------------------------------------------------------------------
    if displayResults:
        imgRoI             = displayRoILines(imgUndist, src_points_list, thickness = 2)
        imgRoIWarped, _, _ = perspectiveTransform(imgRoI , src_points, dst_points, debug = False)
        imgRoIWarped       = displayRoILines(imgRoIWarped, dst_points_list, thickness = 2, color = 'green')
        # print('imgRoI shape       :', imgRoI.shape, imgRoI.min(), imgRoI.max())
        # print('imgRoIWarped shape       :', imgRoIWarped.shape, imgRoIWarped.min(), imgRoIWarped.max())
        display_two(imgRoI  , imgRoIWarped, title1 = 'imgRoI',title2 = ' imgRoIWarped', winttl = filename[0])

    ###----------------------------------------------------------------------------------------------
    ### Display thresholded image before and after Perspective transform WITH RoI line display
    ###----------------------------------------------------------------------------------------------
    # imgThrshldRoI = displayRoILines(imgThrshld, RoI_vertices_list, thickness = 1)
    # imgThrshldRoIWarped, M, Minv = perspectiveTransform(imgThrshldRoI, src_points, dst_points, debug = False)
    # print('imgThrshldRoI shape     :', imgThrshldRoI.shape, imgThrshldRoI.min(), imgThrshldRoI.max())
    # print('img Thrshld Warped shape:', imgThrshldWarped.shape, imgThrshldWarped.min(), imgThrshldWarped.max())
    # display_two(imgThrshldRoI, imgThrshldRoIWarped, title1 = 'imgThrshldRoI',title2 = 'imgThrshldRoIWarped', winttl = filename[0])

    ###----------------------------------------------------------------------------------------------
    ### Display thresholded image without and with RoI line display
    ###----------------------------------------------------------------------------------------------
    # display_two(imgThrshld, imgThrshldRoI, title1 = 'imgThrshld',title2 = 'imgThrshldRoI', winttl = filename[0])

    ###----------------------------------------------------------------------------------------------
    ### Display MASKED color image with non RoI regions masked out -- With RoI line display
    ###----------------------------------------------------------------------------------------------
    # imgMaskedDbg = region_of_interest(imgRoI, RoI_vertices)
    # imgMaskedWarpedDbg, _, _ = perspectiveTransform(imgMaskedDbg, src_points, dst_points, debug = False)
    # print('imgMaskedDebug shape    :', imgMaskedDbg.shape, imgMaskedDbg.min(), imgMaskedDbg.max())
    # print('imgMaskedWarpedDbg shape:', imgMaskedWarpedDbg.shape, imgMaskedWarpedDbg.min(), imgMaskedWarpedDbg.max())
    # display_two(imgMaskedDbg  , imgMaskedWarpedDbg, title1 = 'imgMaskedDebug',title2 = ' imgMaskedWarpedDebug', winttl = filename[0])

    ###----------------------------------------------------------------------------------------------
    ### Display MASKED color image with non RoI regions masked out -- WITHOUT RoI line display
    ###----------------------------------------------------------------------------------------------
    # imgMaskedDbg = region_of_interest(imgUndist, RoI_vertices)
    # imgMaskedWarpedDbg, _, _ = perspectiveTransform(imgMaskedDbg, src_points, dst_points, debug = False)
    # print('imgMaskedDebug shape    :', imgMaskedDbg.shape, imgMaskedDbg.min(), imgMaskedDbg.max())
    # print('imgMaskedWarpedDbg shape:', imgMaskedWarpedDbg.shape, imgMaskedWarpedDbg.min(), imgMaskedWarpedDbg.max())
    # display_two(imgMaskedDbg  , imgMaskedWarpedDbg, title1 = 'imgMaskedDebug',title2 = ' imgMaskedWarpedDebug', winttl = filename[0])

    ###----------------------------------------------------------------------------------------------
    ###  Display Warped --> Thresholded  and Thresholding --> Warped Images
    ###----------------------------------------------------------------------------------------------
    imgWarpedThrshldDict  = apply_thresholds_v1(imgWarped, ksize=ksize, 
                                                  x_thr = x_thr2 , y_thr   = y_thr2, 
                                                mag_thr = mag_thr, dir_thr = dir_thr,
                                                sat_thr = sat_thr, debug   = False)
    imgWarpedThrshld = imgWarpedThrshldDict[thresholdKey]

    ################################################################################################
    ### Select image we want to process further 
    ################################################################################################
    if mode == 1:
        wrk_title = ' Mode 1: imgThrshldWarped : Threshold --> Warp ' 
        working_image = imgThrshldWarped; sfx = '_thr_wrp'   ### Warped AFTER thresholding
    else:
        wrk_title = ' Mode 2: imgWarpedThrshld : Warp --> Thresholding '
        working_image = imgWarpedThrshld; sfx = '_wrp_thr'   ### Warped BEFORE thresholding


    ###----------------------------------------------------------------------------------------------
    ### Display thresholded image befoire and after Perspective transform WITHOUT RoI line display
    ### Display undistorted color image & perpective transformed image -- WITHOUT RoI line display
    ###----------------------------------------------------------------------------------------------
    if displayResults:
        # print('imgUndist shape       :', imgUndist.shape, imgUndist.min(), imgUndist.max())
        # print('imgWarped shape       :', imgWarped.shape, imgWarped.min(), imgWarped.max())
        # print('img Thrshld Warped shape:', imgThrshldWarped.shape, imgThrshldWarped.min(), imgThrshldWarped.max())
        display_two(imgUndist  , imgWarped, 
                    title1 = 'imgUndist - Undistorted Image',
                    title2 = 'imgWarped - Undistorted and Perspective Transformed', winttl = filename[0])
        display_two(imgThrshld, imgThrshldWarped, 
                    title1 = 'imgThrshld - using '+thresholdKey,
                    title2 = 'imgThrshldWarped - Thresholded and Warped image ', winttl = filename[0])
        display_two(imgWarped, imgWarpedThrshld,
                    title1 = 'imgWarped - Warped Image',
                    title2 = 'imgWarpedThrshld - Warped, Thresholded image', winttl = filename[0])
                
        ##----------------------------------------------------------------------------------------------
        # Experiment using histograms on undistorted image to find source x coordinates for  
        ##----------------------------------------------------------------------------------------------
        hist = np.sum(imgThrshldWarped[2*imgThrshldWarped.shape[0]//3:, :], axis=0)
        reg  =  imgThrshldWarped[2*imgThrshldWarped.shape[0]//3:, :]
        # hist, reg, leftx, rightx = pixelHistogram(imgThrshld, y_src_bot-30, y_src_bot+4 , x_src_bot_left - 30, x_src_bot_right + 30)
        display_two(hist, reg, size = (25,5), title1 = ' Mode 1' , title2 = ' img Thresholded --> Warped') 
                                                            
        hist = np.sum(imgWarpedThrshld[2*imgWarpedThrshld.shape[0]//3:, :], axis=0)
        reg  =  imgWarpedThrshld[2*imgWarpedThrshld.shape[0]//3:, :]
        # hist, reg, leftx, rightx = pixelHistogram(imgThrshld, y_src_top-4 , y_src_top+20, x_src_top_left - 30, x_src_top_right + 30)
        display_two(hist, reg, size = (25,5), title1 = ' Mode 2',  title2 = ' img  Warped --> Thresholded') 

        # display_one(working_image, title = wrk_title)


        

    leftx, lefty, rightx, righty, out_img, histogram = find_lane_pixels_v1(working_image,
                                                                            nwindows = nwindows,
                                                                            window_margin = window_margin,
                                                                            debug = debug2)
   
    imgLanePixels = colorLanePixels_v1(out_img, leftx, lefty, rightx, righty)
    
    left_fit, right_fit = fit_polynomial_v1(leftx, lefty, rightx, righty);

    ploty, left_fitx, right_fitx = plot_polynomial_v1(imgLanePixels.shape[0], left_fit, right_fit)

    result_1 = displayLaneRegion_v1(imgUndist, left_fitx, right_fitx, Minv)

    curv_msg, _ , _ , _  = measure_curvature(700, left_fit, right_fit, 'm', debug = False)

    result_2 = displayText(result_1, 40,40, curv_msg, fontHeight = 25)

    oc_msg = offCenterMsg_v1(700, left_fitx[700], right_fitx[700], camera_x, debug = False)

    result_2 = displayText(result_2, 40,80, oc_msg, fontHeight = 25)
    displayGuidelines(result_2, draw = 'y');



    if displayResults:
        print(curv_msg)
        print(oc_msg)

        display_two( imgThrshldWarped, imgWarpedThrshld,
                    title1 = 'Mode 1 - Image Warped AFTER Thresholding',
                    title2 = 'Mode 2 - Image Warped BEFORE Thresholding') 

        disp_curvatures(left_fit, right_fit)
        disp  = PlotDisplay(4,2)
        disp.addPlot(imgLanePixels, title = ' imgLanePixels: Warped image Lane Pixels')
        disp.addPlot(histogram ,type = 'plot' ,  title = ' Histogram of activated pixels')

        # disp2.addPlot(working_image)  ## same as imgWarped
        disp.addPlot(imgLanePixels, title='image Lane Pixels')
        disp.addPlot(left_fitx    , ploty,  subplot = disp.subplot, type = 'plot', color = 'yellow')
        disp.addPlot(right_fitx   , ploty,  subplot = disp.subplot, type = 'plot', color = 'yellow')

        disp.addPlot(result_1, title='result_1 : image Lane Pixels')
        disp.addPlot(result_2, title='result_2 : image Lane Pixels')
        disp.closePlot()
    else:
        disp = None
        
    return result_2, disp


def saveOutputImage( inputFilename, result, sfx = '', outputPath =  './output_images/',  mode = None):
    assert mode is not None  and (0 < mode < 3) , ' mode must be 1 or 2'
    if mode == 1:
        # print(' Mode 1: Threshold Image --> Warp ' )
        sfx1 = '_mode1'   ### Warped AFTER thresholding
    else:
        # print(' Mode 2: Warp Image  --> Thresholding ' )
        sfx1 = '_mode2'   ### Warped BEFORE thresholding

    in_filename_ext = os.path.basename(inputFilename)
    in_filename     = os.path.splitext(in_filename_ext)
    out_filename    = in_filename[0]+'_output'+sfx1+sfx+in_filename[1]
    save_filename   = outputPath+out_filename
    display_one(result, winttl = save_filename, title = in_filename_ext +' - Final Result')
    try:
        rc = mpimg.imsave(save_filename, result)
    except Exception as e:
        print(' image save failed  ')
        print(e)
    else:
        print(' Saved to : ' , save_filename)


def loadCameraCalibration(configFilename):
    with open(configFilename, 'rb') as infile:
        camera = pickle.load(infile)
        print(' Camera calibration file loaded ...')
        
    print()
    print(' Camera Calibration Matrix :')
    print(' ---------------------------')
    print(camera.cameraMatrix)
    return camera

    
def disp_curvatures(left_fit, right_fit):
    print('radius of curvature')
    print()
    print('-'*80)
    print("  {:8s}  {:8s}   {:8s}   {:8s}      {:8s}   {:8s}  {:8s} ".format(" y eval" ,"avg pxl", "left_pxl" , "right_pxl", "avg mtr","left_mtr", "right_mtr"))
    print('-'*80)
    for y_eval in range(20,730,50):
        msg, curv_avg_pxl, curv_left_pxl, curv_right_pxl = measure_curvature(y_eval, left_fit, right_fit, 'p')
        msg, curv_avg_mtr, curv_left_mtr, curv_right_mtr = measure_curvature(y_eval, left_fit, right_fit, 'm')
        print(" {:8.0f}   {:8.2f}   {:8.2f}   {:8.2f}      {:8.2f}   {:8.2f}   {:8.2f} ".format(y_eval, 
                                                                                                curv_avg_pxl, curv_left_pxl, curv_right_pxl, 
                                                                                                curv_avg_mtr, curv_left_mtr, curv_right_mtr))
     
"""
def run_pipeline_on_video_frame(pipelineFunction, videoFile, frameId = 0 , debug = True):
    '''
    Apply pipeline processing on a single frame of a video stream
    
    pipeline_function:  Name of pipeline function to use for processing
    videoFile        :  Video file to process
    frame_id         :  Frame number to process
    
    '''
    cap = cv2.VideoCapture(videoFile) 
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    print()
    print(' Information on :', videoFile)
    print(' ','-'*(18+len(videoFile)))
    print(' Next_frame: ', cap.get(1), ' time(ms): ', cap.get(0),' Rel pos: ',cap.get(2),' Width: ',cap.get(3),' Height: ',cap.get(4), 'Frame Rate: ',
          cap.get(5),      ' Codec: ', cap.get(6), ' Ttl number of frames: ', cap.get(7))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frameId)
    print(' Next frame to read set to : ', cap.get(cv2.CAP_PROP_POS_FRAMES))

    ret, frameBGR = cap.read()

    frameRGB = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2RGB)
    next_frame = cap.get(1)
    print(' Current frame: ', next_frame-1, ' curr pos(ms): ', cap.get(0), 'Frame Rate: ',cap.get(5), ' Ttl frames: ', cap.get(7))
    print()
    
    result = pipelineFunction(frameRGB, debug= True)

    fig = plt.figure(figsize=(15,7))
    plt.imshow(result); plt.title(' Final Result - Frame: '+str(next_frame-1)+'     Curr pos(ms): '+ str(round(cap.get(0),1)) ); plt.show()

    cap.release()
    
    return result

def run_pipeline_on_video_frame_range(pipelineFunction, videoFile, fromFrame = 0 , toFrame = 0, debug = False):
    '''
    Apply pipeline processing on a range of frames  of a video stream
    
    pipeline_function:  Name of pipeline function to use for processing
    videoFile        :  Video file to process
    fromFrame        :  Starting frame number 
    toFrame          :  Ending frame
    '''
    cap = cv2.VideoCapture(videoFile)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    cap.set(cv2.CAP_PROP_POS_FRAMES, fromFrame)
    
    nextFrame = fromFrame
    
    while cap.isOpened() and (nextFrame <= toFrame) :
        ret, frameBGR = cap.read()
        if frameBGR is None:
            break
        frameRGB = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2RGB)
        
        nextFrame = cap.get(1)
        imgDetected = pipelineFunction(frameRGB, debug= False)

        fig = plt.figure(figsize=(10,5))
        plt.imshow(imgDetected); plt.title(' Frame: '+str(nextFrame-1)+'     Curr pos(ms): '+ str(round(cap.get(0),1)) + '     Rel pos: '+ str(cap.get(2))); plt.show()


def save_video_output_frame( img, frameNum,  videoFilePath, videoFile, ext = '.jpg'):
    '''
    save a processed video frame using the video clip name suffixed
    with the frame number:    {video_file_name}_frame_{frameNum}.jpg
    '''
    filename_ext = os.path.basename(videoFile)
    filename = os.path.splitext(filename_ext)
    output_filename = os.path.join(videoFilePath, filename[0]+'_frame_'+str(int(frameNum))+ext)
    mpimg.imsave(output_filename, img)
    
    # print(' Basename: ', filename_ext, '  filename ' , filename) 
    print(' Frame written to : ', output_filename)
    



"""
