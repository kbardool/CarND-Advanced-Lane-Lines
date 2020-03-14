
'''
    def find_lane_pixels(self, binary_warped, LLane, RLane, histRange = None,
                     nwindows = 9, 
                     window_margin   = 100, 
                     minpix   = 90,
                     maxpix   = 0, 
                     debug    = False):

        LLane.set_height(binary_warped.shape[0])
        RLane.set_height(binary_warped.shape[0])

        if histRange is None:
            histLeft = 0
            histRight = binary_warped.shape[1]
        else:
            histLeft, histRight = int(histRange[0]), int(histRange[1])
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped, np.ones_like(binary_warped)*255))
        
        # if debug: 
            # print(' binary warped shape: ', binary_warped.shape)
            # print(' out_img shape: ', out_img.shape)
            # display_one(out_img, grayscale = False, title = 'out_img')
            # display_one(binary_warped, title='binary_warped')

        # Take a histogram of the bottom half of the image
        
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, histLeft:histRight], axis=0)
        print(' histogram shape before padding: ' , histogram.shape)
        
        histogram = np.pad(histogram, (histLeft, binary_warped.shape[1]-histRight))
        print(' histogram shape after padding : ' , histogram.shape) 
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        
        midpoint    = np.int(histogram.shape[0]//2)
        leftx_base  = np.argmax(histogram[:midpoint]) 
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint     
        
        if debug:
            print(' Run find_lane_pixels()  - histRange:', histRange)
            print(' Midpoint:  {} '.format(midpoint))
            print(' Histogram left side max: {}  right side max: {}'.format(np.argmax(histogram[:midpoint]),np.argmax(histogram[midpoint:])))
            print(' Histogram left side max: {}  right side max: {}'.format(leftx_base, rightx_base))
        
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        # nwindows = 9
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Set the width of the windows +/- margin
        # margin = 100
        # Set minimum number of pixels found to recenter window
        # minpix = 90
        # Set maximum number of pixels found to recenter window
        if maxpix == 0 :
            maxpix = (window_height * window_margin )
        
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
      
        # Current positions to be updated later for each window in nwindows
        leftx_current  = leftx_base
        rightx_current = rightx_base
        
        if debug:
            print(' left x base  : ', leftx_base, '  right x base :', rightx_base )
            print(' window_height: ', window_height)
            print(' nonzero x    : ', nonzerox.shape, nonzerox)
            print(' nonzero y    : ', nonzeroy.shape, nonzeroy)
            print(' Starting Positions: left x :', leftx_current, '  right x: ', rightx_current )
        
        # Create empty lists to receive left and right lane pixel indices
        left_line_inds  = []
        right_line_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low  = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low   = leftx_current  - window_margin  
            win_xleft_high  = leftx_current  + window_margin  
            win_xright_low  = rightx_current - window_margin       
            win_xright_high = rightx_current + window_margin  

            if debug:
                print()
                print(' Window: ', window, ' y range: ', win_y_low,' to ', win_y_high )
                print('-'*50)
                print(' Left  lane X range : ', win_xleft_low , '  to  ', win_xleft_high)
                print(' Right lane X range : ', win_xright_low, '  to  ', win_xright_high)
                
            # Draw the windows on the visualization image
            window_color = colors.to_rgba('green')
            cv2.rectangle(out_img,(win_xleft_low , win_y_low), (win_xleft_high , win_y_high), window_color, 2) 
            cv2.rectangle(out_img,(win_xright_low, win_y_low), (win_xright_high, win_y_high), window_color, 2) 
            
            ### MY SOLUTION: Identify the nonzero pixels in x and y within the window -------------
            left_x_inds = np.where((win_xleft_low <=  nonzerox) & (nonzerox < win_xleft_high))
            left_y_inds = np.where((win_y_low     <=  nonzeroy) & (nonzeroy < win_y_high))
            good_left_inds = np.intersect1d(left_x_inds,left_y_inds,assume_unique=False)
            
            right_x_inds = np.where((win_xright_low <= nonzerox) & (nonzerox < win_xright_high))
            right_y_inds = np.where((win_y_low     <=  nonzeroy) & (nonzeroy < win_y_high))
            good_right_inds = np.intersect1d(right_x_inds,right_y_inds,assume_unique=False)
            ###------------------------------------------------------------------------------------

            ### UDACITY SOLUTION: Identify the nonzero pixels in x and y within the window ###
            # good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            # (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            # good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            # (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            ###------------------------------------------------------------------------------------
            
            if debug:
                print()
                print(' left_x_inds  : ', left_x_inds[0].shape, ' left_y_indx  : ', left_y_inds[0].shape,
                      ' -- good left inds size: ', good_left_inds.shape[0])
                # print(' X: ', nonzerox[good_left_inds]) ; print(' Y: ', nonzeroy[good_left_inds])
                print(' right_x_inds : ', right_x_inds[0].shape, ' right_y_indx : ', right_y_inds[0].shape,
                      '  -- good right inds size: ', good_right_inds.shape[0])
                # print(' X: ', nonzerox[good_right_inds]); print(' Y: ', nonzeroy[good_right_inds])
            
            # Append these indices to the lists
            left_line_inds.append(good_left_inds)
            right_line_inds.append(good_right_inds)
            
            ### If #pixels found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position    ###
            if (maxpix > good_left_inds.shape[0] > minpix):
                left_msg  = ' Set leftx_current  :  {} ---> {} '.format( leftx_current,  int(nonzerox[good_left_inds].mean()))
                leftx_current = int(nonzerox[good_left_inds].mean())
            else:
                left_msg  = ' Keep leftx_current :  {} '.format(leftx_current)

            if (maxpix > good_right_inds.shape[0] > minpix ) :
                right_msg = ' Set rightx_current :  {} ---> {} '.format( rightx_current, int(nonzerox[good_right_inds].mean()))
                rightx_current = int(nonzerox[good_right_inds].mean())
            else:
                right_msg = ' Keep rightx_current:  {} '.format(rightx_current)
            
            if debug:
                print(left_msg)
                print(right_msg)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_line_inds  = np.concatenate(left_line_inds)
            right_line_inds = np.concatenate(right_line_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            print(' concatenate not working ')
            pass
        
        # Extract left and right line pixel positions
        LLane.set_linePixels(nonzerox[left_line_inds], nonzeroy[left_line_inds])
        RLane.set_linePixels(nonzerox[right_line_inds], nonzeroy[right_line_inds])
        
        if debug:
            print()
            print(' leftx : ', LLane.allx.shape, ' lefty : ', LLane.ally.shape)
            # print('   X:', LLane.allx[:15])
            # print('   Y:', LLane.ally[:15])
            # print('   X:', LLane.allx[-15:])
            # print('   Y:', LLane.ally[-15:])
            print(' rightx : ', RLane.allx.shape, ' righty : ', RLane.ally.shape)
            # print('   X:', RLane.allx[:15])
            # print('   Y:', RLane.ally[:15])
            # print('   X:', RLane.allx[-15:])
            # print('   Y:', RLane.ally[-15:])
            # display_one(out_img)

        return out_img, histogram 

'''

'''


    def search_around_poly(self, binary_warped, LLane, RLane, search_margin = 100, debug = False):
        """
        # HYPERPARAMETER
        # search_margin : width of the margin around the previous polynomial to search
        """
        out_img = np.dstack((binary_warped, binary_warped, binary_warped, np.ones_like(binary_warped)))*255

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Grab activated pixels
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        # left_fit  = LLane.current_fit
        # right_fit = RLane.current_fit

        left_fit  = LLane.best_fit
        right_fit = RLane.best_fit

        fitted_x_left     = (left_fit [0]*nonzeroy**2) + ( left_fit[1]*nonzeroy) + left_fit[2]
        fitted_x_right    = (right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + right_fit[2]
            
        left_line_inds  = ( (nonzerox > ( fitted_x_left - search_margin )) & (nonzerox < ( fitted_x_left + search_margin)) ).nonzero()
        right_line_inds = ( (nonzerox > (fitted_x_right - search_margin)) & (nonzerox <  (fitted_x_right + search_margin)) ).nonzero()
        
        if debug:
            print(' Search_around_poly() ')
            print(' fitted_x_left  : ', fitted_x_left.shape     , '  fitted_x_right : ', fitted_x_right.shape)
            print(' left_lane_inds : ',  left_line_inds[0].shape , left_line_inds)
            print(' right_lane_inds: ', right_line_inds[0].shape, right_line_inds)
        
        
        # Extract left and right line pixel positions
        # LLane.allx = nonzerox [left_line_inds]
        # LLane.ally = nonzeroy [left_line_inds] 
        # RLane.allx = nonzerox[right_line_inds]
        # RLane.ally = nonzeroy[right_line_inds]
        LLane.set_linePixels(nonzerox [left_line_inds],  nonzeroy[left_line_inds])
        RLane.set_linePixels(nonzerox[right_line_inds], nonzeroy[right_line_inds])
        
        # Fit new polynomials
        # left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        out_img= displayPolySearchRegion(out_img, LLane, RLane, debug = debug)
        
        return out_img, histogram
      
        # return result

'''

'''
    def find_lane_pixels_v1(self, binary_warped, histRange= None,  debug = False):
        self.LeftLane.height  = binary_warped.shape[0]
        self.RightLane.height = binary_warped.shape[0]

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        # if debug: 
            # print(' binary warped shape: ', binary_warped.shape)
            # print(' out_img shape: ', out_img.shape)
            # display_one(out_img, grayscale = False, title = 'out_img')
            # display_one(binary_warped, title='binary_warped')
        if histRange is None:
            histLeft = 0
            histRight = binary_warped.shape[1]
        else:
            histLeft, histRight = int(histRange[0]), int(histRange[1])

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,histLeft:histRight], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint    = np.int(histogram.shape[0]//2)
        leftx_base  = np.argmax(histogram[:midpoint]) + histLeft
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint + histLeft

        if debug:
            print(' Hist range left: {}  right: {} '.format(histLeft,histRight))
            print(' Midpoint:  {} '.format(midpoint))
            print(' Histogram left side max: {}  right side max: {}'.format(np.argmax(histogram[:midpoint]),np.argmax(histogram[midpoint:])))
            print(' Histogram left side max: {}  right side max: {}'.format(leftx_base, rightx_base))
        
        # HYPERPARAMETERS
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.NWINDOWS)
        # Set maximum number of pixels found to recenter window
        self.MAXPIX = (window_height * self.WINDOW_MARGIN)

        # Choose the number of sliding windows
        # NWINDOWS = 9
        # Set the width of the windows +/- margin
        # MARGIN = 100
        # Set minimum number of pixels found to recenter window
        # MINPIX = 90
        
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
      
        # Current positions to be updated later for each window in nwindows
        leftx_current  = leftx_base
        rightx_current = rightx_base

        if debug:
            print(' left x base  : ', leftx_base, '  right x base :', rightx_base )
            print(' window_height: ', window_height)
            print(' nonzero x    : ', nonzerox.shape, nonzerox)
            print(' nonzero y    : ', nonzeroy.shape, nonzeroy)
            print(' Starting Positions: left x :', leftx_current, '  right x: ', rightx_current )
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds  = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.NWINDOWS):
            # Identify window boundaries in x and y (and right and left)
            win_y_low  = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low   = leftx_current  - self.WINDOW_MARGIN # Update this
            win_xleft_high  = leftx_current  + self.WINDOW_MARGIN # Update this
            win_xright_low  = rightx_current - self.WINDOW_MARGIN # Update this     
            win_xright_high = rightx_current + self.WINDOW_MARGIN # Update this

            if debug:
                print()
                print(' Window: ', window, ' y range: ', win_y_low,' to ', win_y_high )
                print('-'*50)
                print(' Left  lane X range : ', win_xleft_low , '  to  ', win_xleft_high)
                print(' Right lane X range : ', win_xright_low, '  to  ', win_xright_high)
                
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low , win_y_low), (win_xleft_high , win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low, win_y_low), (win_xright_high, win_y_high),(0,255,0), 2) 
            
            ### MY SOLUTION: Identify the nonzero pixels in x and y within the window -------------
            left_x_inds = np.where((win_xleft_low <=  nonzerox) & (nonzerox < win_xleft_high))
            left_y_inds = np.where((win_y_low <=  nonzeroy) & (nonzeroy < win_y_high))
            good_left_inds = np.intersect1d(left_x_inds,left_y_inds,assume_unique=False)
            
            right_x_inds = np.where((win_xright_low <= nonzerox) & (nonzerox < win_xright_high))
            right_y_inds = np.where((win_y_low <=  nonzeroy) & (nonzeroy < win_y_high))
            good_right_inds = np.intersect1d(right_x_inds,right_y_inds,assume_unique=False)
            ###------------------------------------------------------------------------------------

            ### UDACITY SOLUTION: Identify the nonzero pixels in x and y within the window ###
            # good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            # (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            # good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            # (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            if debug:
                print()
                print(' left_x_inds  : ', left_x_inds[0].shape, ' left_y_indx  : ', left_y_inds[0].shape) 
                print(' good left inds size: ', good_left_inds.shape[0])
                # print(' X: ', nonzerox[good_left_inds]) ; print(' Y: ', nonzeroy[good_left_inds])
                print(' right_x_inds : ', right_x_inds[0].shape, ' right_y_indx : ', right_y_inds[0].shape)  
                print(' good right inds size: ', good_right_inds.shape[0])
                # print(' X: ', nonzerox[good_right_inds]); print(' Y: ', nonzeroy[good_right_inds])
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### If #pixels found > MINPIX pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position    ###
            if (self.MAXPIX > good_left_inds.shape[0] > self.MINPIX):
                left_msg  = ' Set leftx_current  :  {} ---> {} '.format( leftx_current,  int(nonzerox[good_left_inds].mean()))
                leftx_current = int(nonzerox[good_left_inds].mean())
            else:
                left_msg  = ' Keep leftx_current :  {} '.format(leftx_current)

            if (self.MAXPIX > good_right_inds.shape[0] > self.MINPIX ) :
                right_msg = ' Set rightx_current :  {} ---> {} '.format( rightx_current, int(nonzerox[good_right_inds].mean()))
                rightx_current = int(nonzerox[good_right_inds].mean())
            else:
                right_msg = ' Keep rightx_current:  {} '.format(rightx_current)
            
            if debug:
                print(left_msg)
                print(right_msg)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds  = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            print(' concatenate not working ')
            pass
        
        # Extract left and right line pixel positions
        self.LeftLane.allx  = nonzerox[left_lane_inds]
        self.LeftLane.ally  = nonzeroy[left_lane_inds] 
        self.RightLane.allx = nonzerox[right_lane_inds]
        self.RightLane.ally = nonzeroy[right_lane_inds]
        
        if debug:
            print()
            print(' leftx : ', self.LeftLane.allx.shape, ' lefty : ', self.LeftLane.ally.shape)
            print('   X:', self.LeftLane.allx[:15])
            print('   Y:', self.LeftLane.ally[:15])
            print('   X:', self.LeftLane.allx[-15:])
            print('   Y:', self.LeftLane.ally[-15:])
            print(' rightx : ', self.RightLane.allx.shape, ' righty : ', self.RightLane.ally.shape)
            print('   X:', self.RightLane.allx[:15])
            print('   Y:', self.RightLane.ally[:15])
            print('   X:', self.RightLane.allx[-15:])
            print('   Y:', self.RightLane.ally[-15:])

        self.processSlidingWin = True

        return out_img, histogram
'''

'''             
    def search_around_poly_v1(self, binary_warped, debug = False):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # MARGIN = 100

        # Grab activated pixels
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        # left_fit = self.LeftLane.current_fit
        # right_fit = self.RightLane.current_fit
        left_fit  = self.LeftLane.best_fit
        right_fit = self.RightLane.best_fit
        
        fitted_x_left     = (left_fit[0] * nonzeroy**2) + (left_fit[1] * nonzeroy) + (left_fit[2])
        fitted_x_right    = (right_fit[0]* nonzeroy**2) + (right_fit[1]* nonzeroy) + (right_fit[2])
            
        left_lane_inds  = ((nonzerox > fitted_x_left  - self.WINDOW_MARGIN) & (nonzerox < fitted_x_left  + self.WINDOW_MARGIN)).nonzero()
        right_lane_inds = ((nonzerox > fitted_x_right - self.WINDOW_MARGIN) & (nonzerox < fitted_x_right + self.WINDOW_MARGIN)).nonzero()
        
        if debug:
            print(' Search_around_poly() ')
            print(' fitted_x_left  : ', fitted_x_left.shape     , '  fitted_x_right : ', fitted_x_right.shape)
            print(' left_lane_inds : ', left_lane_inds[0].shape , left_lane_inds)
            print(' right_lane_inds: ', right_lane_inds[0].shape, right_lane_inds)
        
        # Extract left and right line pixel positions
        self.LeftLane.allx  = nonzerox[left_lane_inds]
        self.LeftLane.ally  = nonzeroy[left_lane_inds] 
        self.RightLane.allx = nonzerox[right_lane_inds]
        self.RightLane.ally = nonzeroy[right_lane_inds]
        
        return out_img, histogram
'''


"""    
def displayPolySearchRegion2(input_img, left_fit, right_fit, margin = 100,  debug = False):
    color = colors.to_rgba('springgreen')

    # Generate y values for plotting
    height = input_img.shape[0]
    ploty = np.linspace(0, height-1, height)

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(input_img)
    
    try:
        left_fitx   = (left_fit[0] * ploty**2) + (left_fit[1] * ploty) + left_fit[2]
        right_fitx  = (right_fit[0]* ploty**2) + (right_fit[1]* ploty) + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx  = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts     = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts     = np.hstack((right_line_window1, right_line_window2))

    if debug:
        print('displayPolySearchRegion2() ')
        print(' left fit parms : ', left_fit)
        print(' right_fit_parms: ', right_fit)
        print(left_line_pts.shape, right_line_pts.shape)
        
    # Draw the search region onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), color)
    
    result = cv2.addWeighted(input_img, 1, window_img, 0.6, 0)
    
    return result   
  
  
def displayPolynomial2(input_img, left_fitx, right_fitx, color = 'red', debug = False):

    print('displayPolynomial2 : ', input_img.shape)    
    # Generate y values for plotting

    left_height  = left_fitx.shape[0]
    right_height = right_fitx.shape[0]
    
    left_ploty   = (np.linspace(0, left_height-1, left_height, dtype = np.int))
    right_ploty  = (np.linspace(0, right_height-1, right_height, dtype = np.int))
    
    colorRGBA = colors.to_rgba(color)
    result = np.copy(input_img)
    result[left_ploty, np.int_(np.round_(left_fitx ,0))] = colorRGBA
    result[left_ploty, np.int_(np.round_(right_fitx,0))] = colorRGBA
    
    return  result 
 
    
def displayDetectedRegion1(input_img, LLane, RLane,  Minv, **kwargs):
    '''
    iteration: item from xfitted_history to use for lane region zoning
               -1 : most recent xfitted current_xfitted (==  xfitted_history[-1])
    '''
    beta  = kwargs.setdefault( 'beta', 0.5) 
    start = kwargs.setdefault('start', 0)  
    debug = kwargs.setdefault('debug', False)
    iteration = kwargs.setdefault('iteration', -1)
   
    left_ploty  = LLane.ploty[start:]
    left_fitx   = LLane.xfitted_history[iteration][start:]
    right_ploty = RLane.ploty[start:]
    right_fitx  = RLane.xfitted_history[iteration][start:]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(input_img).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, left_ploty]))]).astype(np.int)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_ploty])))]).astype(np.int)
    pts       = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, ([pts]), (0,255, 0))
    
    # draw left and right lanes
    cv2.polylines(color_warp, (pts_left) , False, (255,0,0), thickness=18, lineType = cv2.LINE_AA)
    cv2.polylines(color_warp, (pts_right), False, (0,0,255), thickness=18, lineType = cv2.LINE_AA)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (input_img.shape[1], input_img.shape[0])) 
    

    if debug:
        print('undistImage : ', input_img.shape, ' newwarp : ', newwarp.shape)
    
    # Combine the result with the original image
    result = cv2.addWeighted(input_img, 1, newwarp, beta, 0)
    
    return result    


def displayPolynomial(input_img, left_fit, right_fit, color = [255,255,0], debug = False):
    # Generate y values for plotting

    height    = input_img.shape[0]
    ploty     = (np.linspace(0, height-1, height, dtype = np.int))
    print(ploty)
    left_fit  = LLane.current_xfitted
    right_fit = RLane.best_fit
    try:
        left_fitx  = np.int_(np.round_(left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] ,0))
        right_fitx = np.int_(np.round_(right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] ,0))
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx  = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
    result = np.copy(input_img)
    result[ploty, left_fitx ] = color
    result[ploty, right_fitx] = color
    
    return  result 
    
    
def find_lanes_and_fit_polynomial(binary_warped, debug = False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    left_fit, right_fit          = fit_polynomial(leftx, lefty, rightx, righty, out_img)
    ploty, left_fitx, right_fitx = plot_polynomial(binary_warped.shape[0], left_fit, right_fit)

    output_img  = display_lane_pixels(out_img, leftx, lefty, rightx, righty)
    
    return ploty, left_fitx, right_fitx, output_img


def colorLanePixels(input_img, LeftLane, RightLane, debug = False):
    ## Visualization ##
    # Colors in the left and right lane regions
    if debug: 
        print(' Call displayLanePixels')
    result = np.copy(input_img)
    result[LeftLane.ally, LeftLane.allx]   = [255, 0, 0]
    result[RightLane.ally, RightLane.allx] = [0, 0, 255]    
    return result
"""
    


    

# displayPolynomial()----------------------------------------------------------------------
#
# try:
#     left_fitx   = LLane.xfitted_history[iteration][0,start:]
#     right_fitx  = RLane.xfitted_history[iteration][0,start:]
#     left_ploty  = LLane.xfitted_history[iteration][1,start:]
#     right_ploty = RLane.xfitted_history[iteration][1,start:]
# except:
#     # print(' displayPolynomial() w/ ITERATION=', iteration, 'DOESNT EXIST - IGNORED ')
#     return input_img
# left_fitx    = np.int_(LLane[0,start:])
# left_ploty   = np.int_(LLane[1,start:])    
# right_fitx   = np.int_(RLane[0,start:])
# right_ploty  = np.int_(RLane[1,start:])    
# try:
#     left_fitx    = LLane[iteration][start:]
#     right_fitx   = RLane[iteration][start:]    
#     left_height  = LLane[iteration].shape[0]
#     right_height = RLane[iteration].shape[0] 
#     left_ploty   = np.linspace(start,  left_height-1,  left_height-start, dtype = np.int)
#     right_ploty  = np.linspace(start, right_height-1, right_height-start, dtype = np.int)
# except:
#     # print(' displayPolynomial() w/ ITERATION=', iteration, 'DOESNT EXIST - IGNORED ')
#     return input_img
#---------------------------------------------------------------------------------------------

# displayDetectedRegion() --------------------------------------------------------------------
# color_warp = np.zeros_like(input_img)
# left_idx  = (end > LLane[1,:]) & (LLane[1,:] >= start) 
# right_idx = (end > RLane[1,:]) & (RLane[1,:] >= start) 
# left_x    = np.int_(LLane[0,left_idx])
# left_y    = np.int_(LLane[1,left_idx])
# right_x   = np.int_(RLane[0,right_idx])
# right_y   = np.int_(RLane[1,right_idx])

# Recast the x and y points into usable format for cv2.fillPoly()
# pts_left  = np.array([np.transpose(np.vstack([left_x, left_y]))]).astype(np.int)
# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))]).astype(np.int)
# pts       = np.hstack((pts_left, pts_right))
# left_idx  = (end > LLane[1,:]) & (LLane[1,:] >= start) 
# right_idx = (end > RLane[1,:]) & (RLane[1,:] >= start) 
# pts_left  = np.expand_dims(LLane[:, left_idx].T.astype(np.int), 0)
# pts_right = np.expand_dims(np.flipud(RLane[:,right_idx].T.astype(np.int)),0)
# pts       = np.hstack((pts_left, pts_right))
## - Create blank 2D overlay image 
## - Draw region between detected lanes on blank image 
## - Draw left and right lines with distinctive color
## -  Warp the blank back to original image space using inverse perspective matrix (Minv)
# cv2.fillPoly(color_warp, ([pts]), region_color)
# cv2.polylines(color_warp, ( pts_left), False, (255,0,0), thickness=18, lineType = cv2.LINE_AA)
# cv2.polylines(color_warp, (pts_right), False, (0,0,255), thickness=18, lineType = cv2.LINE_AA)
# newwarp = cv2.warpPerspective(color_warp, Minv, (input_img.shape[1], input_img.shape[0])) 

# left_idx  = (end > LLane[1,:]) & (LLane[1,:] >= start) 
# right_idx = (end > RLane[1,:]) & (RLane[1,:] >= start) 
# pts_left  = np.expand_dims(LLane[:, left_idx].T.astype(np.int), 0)
# pts_right = np.expand_dims(np.flipud(RLane[:,right_idx].T.astype(np.int)),0)
# pts       = np.hstack((pts_left, pts_right))
#--------------------------------------------------------------------------------------------------

# displayText() -----------------------------------------------------------------------------------
# fontFace   = cv2.FONT_HERSHEY_SIMPLEX
# lineType   = cv2.LINE_AA
# output = np.copy(img)
# retval, baseLine= cv2.getTextSize( text, fontFace, fontScale, thickness = thickness)
# baseLine+ fontHeight
# if debug:
    # print('font scale: ', fontScale, ' retval : ', etval, 'Baseline: ', baseLine)
# cv2.putText(output, text, (x, y),fontFace, fontScale, color , thickness, lineType)
#--------------------------------------------------------------------------------------------------



"""
    ## FROM FIND_LANE_PIXELS()

        # if good_right_inds.shape[0] > 0 :
            # min_left  = np.round( nonzerox[good_left_inds].min(),0); min_right  = np.round(nonzerox[good_right_inds].min(),0) 
            # max_left  = np.round( nonzerox[good_left_inds].max(),0); max_right  = np.round(nonzerox[good_right_inds].max(),0)
            # mean_left = np.round(nonzerox[good_left_inds].mean(),0); mean_right = np.round(nonzerox[good_right_inds].mean(),0)
            # med_left  = np.round(np.median(nonzerox[good_left_inds]),0); med_right = np.round(np.median(nonzerox[good_right_inds]),0)
            # Left_min_to_win = (min_left - win_xleft_low) 
            # Left_max_to_win = (win_xleft_high - max_left)
        
            # if Left_min_to_win > Left_max_to_win:
            #     Left_adj  =  shift_amount  ## Left_min_to_win 
            # else:
            #     Left_adj  =  -shift_amount ## -Left_max_to_win
        
            # Left_lower_margin  = window_margin - Left_adj
            # Left_higher_margin = window_margin + Left_adj  


            # Right_min_to_win = (min_right - old_win_xright_low) 
            # Right_max_to_win = (old_win_xright_high - max_right)

            # if Right_min_to_win > Right_max_to_win:
            #     Right_adj  =  shift_amount   ## Right_min_to_win
            # else:
            #     Right_adj  =  -shift_amount  ## -Right_max_to_win

            # Right_lower_margin  = window_margin - Right_adj
            # Right_higher_margin = window_margin + Right_adj  

            # win_xleft_low   = win_xleft_center  -  Left_lower_margin 
            # win_xleft_high  = win_xleft_center  +  Left_higher_margin
            # win_xright_low  = win_xright_center -  Right_lower_margin      
            # win_xright_high = win_xright_center +  Right_higher_margin            
            
            # win_xleft_low   = leftx_current  - window_margin  
            # win_xleft_high  = leftx_current  + window_margin  
            # win_xright_low  = rightx_current - window_margin       
            # win_xright_high = rightx_current + window_margin  

            # if debug:
                # print()
                # print(' Left Window       | {:4d} - {:4d} - {:4d} |'.format( old_win_xleft_low , old_win_xleft_ctr ,  old_win_xleft_high))
                # print(' Left : mean: {:4.0f} median: {:4.0f}  min: {:4.0f}   max: {:4.0f}   dist to mean:  min: {:4.0f}  max: {:4.0f}'\
                #         ' dist to median: min: {:4.0f}  max: {:4.0f}'.
                #         format(mean_left , med_left, min_left,   max_left, mean_left - min_left, 
                #                 max_left - mean_left, med_left - min_left, max_left - med_left))
                # print(' dist to boundaries: min to old_xleft_low {:4.0f}  max to old_xleft_high: {:4.0f}  left_min_to_win: {} left_max_to_win: {}  adj: {:5d}  Lower margin: {}  Higher margin: {}'.
                        # format((min_left - old_win_xleft_low), (old_win_xleft_high - max_left),  Left_min_to_win, Left_max_to_win, Left_adj , Left_lower_margin, Left_higher_margin))
                # print(' mean: {}  percentile:{}  '.format(mean_left, np.percentile(nonzerox[good_left_inds], [25,50, 75])))
                # print(' Left Window       | {:4d} - {:4d} - {:4d} |            Next Left Window   | {:4d} - {:4d} - {:4d} |'.format( 
                        # old_win_xleft_low , old_win_xleft_ctr ,  old_win_xleft_high, win_xleft_low , win_xleft_center ,  win_xleft_high))

                # print()
                # print(' Right Window      | {:4d} - {:4d} - {:4d} |'.format(old_win_xright_low, old_win_xright_ctr, old_win_xright_high))
                # print(' Right: mean: {:4.0f} median: {:4.0f}  min: {:4.0f}   max: {:4.0f}   dist to mean: min: {:4.0f}  max: {:4.0f} '\
                #         ' dist to median: min: {:4.0f}  max: {:4.0f}'.
                #         format(mean_right, med_right, min_right, max_right, mean_right - min_right, 
                #                 max_right - mean_right,med_right - min_right, max_right - med_right))       
                # print(' dist to boundaries: min to left {:4.0f}  max to right: {:4.0f}  R_min_to_win: {} R_max_to_win: {} adj: {:5d}  Lower margin: {}  Higher margin: {}'.
                #         format((min_right - old_win_xright_low), (old_win_xright_high - max_right), Right_min_to_win, Right_max_to_win,  Right_adj , Right_lower_margin, Right_higher_margin))

                # print(' mean: {}  percentile:{}  '.format(mean_right, np.percentile(nonzerox[good_right_inds], [25,50, 75])))
                
                # print(' Right Window      | {:4d} - {:4d} - {:4d} |            Next Right Window  | {:4d} - {:4d} - {:4d} |'.format(
                        # old_win_xright_low, old_win_xright_ctr, old_win_xright_high, win_xright_low, win_xright_center, win_xright_high))
                # print()
        # else:
            # win_xleft_low   = leftx_current  - window_margin  
            # win_xleft_high  = leftx_current  + window_margin  
            # win_xright_low  = rightx_current - window_margin       
            # win_xright_high = rightx_current + window_margin  
            # win_xleft_low   = win_xleft_center  -  Left_lower_margin 
            # win_xleft_high  = win_xleft_center  +  Left_higher_margin
            # win_xright_low  = win_xright_center -  Right_lower_margin      
            # win_xright_high = win_xright_center +  Right_higher_margin
            # if debug:
            #     print()
            #     print(' Left Window       | {:4d} - {:4d} - {:4d} |'.format( old_win_xleft_low , old_win_xleft_ctr ,  old_win_xleft_high))
            #     print(' Next Left Window  | {:4d} - {:4d} - {:4d} |'.format( win_xleft_low , win_xleft_center ,  win_xleft_high))
            #     print()
            #     print(' Right Window      | {:4d} - {:4d} - {:4d} |'.format(old_win_xright_low, old_win_xright_ctr, old_win_xright_high))
            #     print(' Next Right Window | {:4d} - {:4d} - {:4d} |'.format(win_xright_low, win_xright_center, win_xright_high))
            #     print()
"""