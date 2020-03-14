import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from   matplotlib import cm
import copy
from matplotlib.figure import Figure

## Matplotlib backends are:
## ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 
##  'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
##

# Define a help with plotting multiple subplots 

class PlotDisplay(Figure):
    def __init__(self, rows = 1 , columns = 1, width = 25, rowHeight = 7, *args, **kwargs):
        super().__init__( *args, figsize=(width, rows * rowHeight))
        
        # self.oldBackend = copy.copy(mpl.get_backend())
        # plt.switch_backend(backend)
        # self.newBackend = mpl.get_backend()
        # print(' Old backend: ', self.oldBackend, ' New backend: ', self.newBackend)

        self.rows  = rows
        self.cols  = columns 
        self.fig   = plt.figure(figsize=(width, rows * rowHeight))

        self.subplot = 0


    def addPlot(self, *args, 
                subplot= 0, 
                
                **kwargs):
        
        # print(kwargs)
        title = kwargs.setdefault('title' , ' Image ')
        type  = kwargs.setdefault('type'  , 'image')
        xlabel= kwargs.setdefault('xlabel', 'X axis')
        ylabel= kwargs.setdefault('ylabel', 'Y axis')
        color = kwargs.setdefault('color' , 'r')
        cmap  = kwargs.setdefault('cmap' , 'gray')
        grid  = kwargs.setdefault('grid'  , None)

        if self.subplot >= (self.rows * self.cols):
            print(' Maximum number of subplots reached : ',self.rows * self.cols)
            return self.subplot
            
        elif subplot == 0:
            self.subplot += 1
            subplot = self.subplot
        else:
            self.subplot = subplot
            
        ax = self.add_subplot(self.rows, self.cols, subplot)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='k', labelsize = 10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
            
        if type == 'plot':
            # if len(args) == 1:
            ax.plot(*args, color = color)
            # else :
                # ax.plot(args[0], args[1], color = color)
        else:
            ax.imshow(args[0] , cmap=plt.cm.gray)        
            title +='      ' +str(args[0].shape)
        if grid is not None:
            ax.minorticks_on()
            ax.grid(True, which=grid)    
        ax.set_title(title, fontsize=12)
        
        return ax
        
        
    def closePlot(self):
        return self
        # bkend = plt.get_backend()
        # print(' backend : ', bkend)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        # self.canvas.draw() 
        # self.show()
        # plt.switch_backend(self.oldBackend)
    
    def resetPlot(self):
        self.subplot = 0 

'''
    def addPlot(self, img, subplot = 0, title = ' Image ', type = 'image' ):
        if self.subplot >= (self.rows * self.cols):
            print(' Maximum number of subplots reached')
            return self.subplot
            
        elif subplot == 0:
            self.subplot += 1
            subplot = self.subplot
        else:
            self.subplot = subplot
            
        ax = self.add_subplot(self.rows, self.cols, subplot)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='k', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)
        if type == 'plot':
            ax.plot(img)
        else:
            ax.imshow(img , cmap=plt.cm.gray)        
        ax.set_title(title +'      ' +str(img.shape), fontsize=12)
        return ax
'''
        