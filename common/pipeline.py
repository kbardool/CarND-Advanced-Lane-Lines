import os
import matplotlib.image as mpimg


def run_pipeline_on_video_frame(pipelineFunction, videoFile, frameId = 0 , debug = True):
    """
    Apply pipeline processing on a single frame of a video stream
    
    pipeline_function:  Name of pipeline function to use for processing
    videoFile        :  Video file to process
    frame_id         :  Frame number to process
    
    """
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
    """
    Apply pipeline processing on a range of frames  of a video stream
    
    pipeline_function:  Name of pipeline function to use for processing
    videoFile        :  Video file to process
    fromFrame        :  Starting frame number 
    toFrame          :  Ending frame
    """
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
    """
    save a processed video frame using the video clip name suffixed
    with the frame number:    {video_file_name}_frame_{frameNum}.jpg
    """
    filename_ext = os.path.basename(videoFile)
    filename = os.path.splitext(filename_ext)
    output_filename = os.path.join(videoFilePath, filename[0]+'_frame_'+str(int(frameNum))+ext)
    mpimg.imsave(output_filename, img)
    
    # print(' Basename: ', filename_ext, '  filename ' , filename) 
    print(' Frame written to : ', output_filename)
    

    
