import os
import sys
if '..' not in sys.path:
    print("appending '..' to sys.path")
    sys.path.append('..')
import numpy as np
import cv2
import matplotlib.image as mpimg
# from classes.line import Line
# from classes.plotting import PlotDisplay
# from common.utils import (  displayText, displayGuidelines, displayPolySearchRegion)
# from common.sobel import apply_thresholds



class VideoFile(object):

    def __init__(self, videoFilename, mode = 'input', outputPath = './output_videos', debug = False, **kwargs):
        assert mode in ['input', 'output'], "Invalid videoFile mode - must be 'input' or 'output' "
        print()
        print(' ------------------------')
        print(' VideoFile init() routine')
        print(' ------------------------')
        print(' input args: ', kwargs)
        
        
        self.filename      = os.path.basename(videoFilename)
        self.fileBase,self.fileExt   = os.path.splitext(self.filename)
        self.videoFormat   = str.strip(self.fileExt, '.')
        assert self.videoFormat in ['avi', 'mp4'], 'Invalid format type: '+ self.videoFormat

        
        self.videoFile     = None
        self.outputPath    = outputPath
        self.fromFrame     = kwargs.setdefault('fromFrame', 0)
        self.toFrame       = kwargs.setdefault('toFrame'  , None)
        self.suffix        = kwargs.setdefault('suffix'  , '')
        self.like          = kwargs.setdefault('like'  , None)
        self.frameRate     = kwargs.setdefault('frameRate'  , None)
        self.width         = kwargs.setdefault('width'  , None)
        self.height        = kwargs.setdefault('height'  , None)
        self.mode          = mode
        self.currFrameNum  = -1
        self.rangeFinished = False
        
        if self.mode == 'input':
            self.videoFilename = videoFilename
            self._openInputVideoFile()
        else:
            if self.suffix is not '':
                self.suffix = '_'+self.suffix
            self.videoFilename = os.path.join(self.outputPath, self.fileBase)+'_output'+self.suffix+self.fileExt
            if self.like is not None:
                self.frameRate     = self.like.frameRate 
                self.width         = self.like.width     
                self.height        = self.like.height    
            self._openOutputVideoFile()
            
        print(' videoFile _init() complete ',self.videoFilename )
       
    
    
    def _openInputVideoFile(self):
        
        videoFile = cv2.VideoCapture(self.videoFilename)

        if (videoFile.isOpened() == False): 
            print("Error opening video stream or file")
            return -1

        self.videoFile     = videoFile
        self.currPos       = round(videoFile.get(0),2)
        self.currFrameNum  = int(videoFile.get(1)) 
        self.width         = int(videoFile.get(3))
        self.height        = int(videoFile.get(4))
        self.ttlFrames     = int(videoFile.get(7))
        self.frameRate     = videoFile.get(5)
        self.frameTitle    = None

        if self.toFrame is None :
            self.toFrame = self.ttlFrames
        else:
            self.toFrame = min(self.toFrame, self.ttlFrames)
        
        # print()
        # print(' Information on :', self.videoFilename)
        # print(' '+'-'*(18+len(self.videoFilename)))
        # print(' OPEN - get(0) (curr pos): ',self.videoFile.get(0), ' get(1) (next frame):', self.videoFile.get(1), 'get(2) :', self.videoFile.get(2))    
        print(' Width     : ', videoFile.get(3), ' Height  : ', videoFile.get(4), ' Frame Rate: ', round(videoFile.get(5),2),
              ' Codec: ', videoFile.get(6), ' Ttl frames: ', videoFile.get(7))
        print(' FromFrame : ', self.fromFrame  , '   toFrame : ', self.toFrame, '   ttlFrames: ', self.ttlFrames) 

        self.setStartFrame(self.fromFrame)
        # print(' openVideoFile complete  Filename:',self.videoFilename )

    def closeVideoFile(self):
        self.videoFile.release()
        print(' '+self.mode+' video file closed :', self.videoFilename) 
        
    close = closeVideoFile
    
    def _openOutputVideoFile(self):
        
        if self.videoFormat == '.avi':
            codec = cv2.VideoWriter_fourcc('M','J','P','G'),
        else:
            codec = cv2.VideoWriter_fourcc(*'XVID')
            
        self.videoFile = cv2.VideoWriter(self.videoFilename, codec, self.frameRate, (self.width, self.height))
        # print(' opened ' + self.videoFormat  + ' file: ', self.videoFilename)
    
    
    def setFrameRange(self, fromFrame, toFrame):
        self.setStartFrame(fromFrame)
        self.setEndFrame(toFrame)
    
    def setEndFrame(self, toFrame):
        self.toFrame = ttlFrames if toFrame >  self.ttlFrames else toFrame
        self.rangeFinished = False
        print(' Range set from : ',self.fromFrame, ' to : ', self.toFrame , ' Next frame: ',  self.videoFile.get(1), ' ... Reset pipeline')
    
    def setStartFrame(self, fromFrame):
        self.fromFrame = fromFrame
        # print(' Before - get(1) (Next frame)', self.videoFile.get(1), 'currFrameNum: ', self.currFrameNum, ' get(0) currPos(ms): ', self.currPos)
        self.videoFile.set(cv2.CAP_PROP_POS_FRAMES, self.fromFrame)
        
        self.currFrameNum = self.videoFile.get(1) 
        self.currPos      = round(self.videoFile.get(0),2)
        
        # print(' After - get(1) (Next frame)', self.videoFile.get(1), 'currFrameNum: ', self.currFrameNum, ' get(0) currPos(ms): ', self.currPos)
        return fromFrame
        
    def getNextFrame(self, frameNo = None, debug=False):
        if frameNo is not None:
            self.setStartFrame(frameNo)

        if  self.videoFile.get(1) > self.toFrame:
            print(' self.toFrame : ',self.toFrame, ' Next frame: ',  self.videoFile.get(1), ' ... set  rangeFinished to True')
            self.rangeFinished = True
            return False
        
        self.currFrameNum = int(self.videoFile.get(1) )
        
        # print(' Before get(0) (curr pos): ',self.videoFile.get(0), ' get(1) (next frame):', self.videoFile.get(1), 'get(2) :', self.videoFile.get(2))    
        
        ret, imageBGR = self.videoFile.read()
        self.currPos      = round(self.videoFile.get(0),2)
        self.frameTitle   = 'Frame: {:4d}  pos(ms): {:<7.0f}'.format(self.currFrameNum, self.currPos)
        
        # print(' After get(0) (curr pos): ', self.videoFile.get(0), ' get(1) (next frame):', self.videoFile.get(1), 'get(2):', self.videoFile.get(2))    
        
        if imageBGR is None:
            print(' ERROR in reading next frame')
            rc = False 
        else:
            self.image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

            if debug:
                print(' Curr Frame: ', self.currFrameNum , ' Next_frame: ', self.videoFile.get(1), ' time(ms): ', self.currPos)
            if self.toFrame < self.videoFile.get(1):
                if debug:
                    print(' self.rangeFinished : ',self.toFrame, ' Next frame: ',  self.videoFile.get(1), ' ... set  rangeFinished to True')
                self.rangeFinished = True
            rc = True

        return rc



    def saveFrameToImage(self, img, frameNum = None, ext = '.jpg', debug = True):
        """
        save a processed video frame using the video clip name suffixed
        with the frame number:    {video_file_name}_frame_{frameNum}.jpg
        """
     
        output_filename = os.path.join(self.outputPath, self.fileBase+'_frame_'+str(int(frameNum))+self.suffix+ext)
        mpimg.imsave(output_filename, img)
        
        # print(' Basename: ', filename_ext, '  filename ' , filename) 
        if debug:
            print(' Frame written to ' + ext + 'file: ', output_filename)
        
    def saveFrameToVideo(self, frame, debug = True):
        assert self.mode == 'output', ' video file not opened in output mode'
        try:
            frameBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.videoFile.write(frameBGR)
        except Exception as e:
            print('\n ERROR in writing frame', self.currFrameNum, 'to video file')
            print('Exception message:', e)
        else: 
            if debug:
                print(' Frame written to video file: ',self.videoFilename) 
    
   