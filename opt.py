#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# In[10]:


def resize(frame, scale):
    return cv.resize(frame, (int(scale*frame.shape[1]), int(scale*frame.shape[0])))


# In[11]:


cv.destroyAllWindows()


# In[12]:


cap = cv.VideoCapture('IMG_3154.MOV')

ret, frame1 = cap.read()
frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
#plt.imshow(frame1_hsv)
frame1_small = resize(frame1_gray, 0.4)

flow_hsv = np.zeros_like(resize(frame1, 0.4))
flow_hsv[...,1] = 255


# In[ ]:


i = 0
while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    # figure out if it's empty
    frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    #plt.imshow(frame2_hsv)
    frame2_small = resize(frame2_gray,0.4)
    
    flow = cv.calcOpticalFlowFarneback(frame1_small, frame2_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    flow_hsv[...,0] = ang*180/np.pi/2
    flow_hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    
    bgr = cv.cvtColor(flow_hsv,cv.COLOR_HSV2BGR)
    #bgr = resize(bgr, 0.4)
    cv.imshow('flow frame',bgr)    
    cv.imshow('frame', frame2_small)
    
    i += 1
    print(i)


# In[5]:


0/0


# In[ ]:


#print(flow_hsv[0::step,0::step].shape)
#     n, bins, patches = plt.hist(flow_hsv[0::step,0::step,2],bins=20)



# if frame is read correctly ret is True
frame1 = frame2;


cap.release()
cv.destroyAllWindows()


# In[ ]:


step = 5
print(np.reshape(mag,(mag.shape[0]*mag.shape[1])))
print(ang.shape)
#print(flow_hsv[0::step,0::step].shape)

mag[0::step,0::step].shape, ang[0::step,0::step].shape
mag[0::step,0::step][:2, :2]
num, bins = np.histogram2d(mag[0::step,0::step],ang[0::step,0::step])
# #plt.show()
# print(num)
# print(bins.astype(int))


# In[ ]:


plt.axes()


# In[ ]:


import numpy as np
import cv2
import sys
## sys.path.append( r'/home/yakov/Github/opencv/samples/python' )
#sys.path.append( r'C:\Users\yakov\Git\Github\opencv\samples\python' )

#import opt_flow as of

def my_draw_flow(img, flow, **kwargs):

    step = kwargs.get( 'step', 5 )
    gr_thresh = kwargs.get( 'gr_thresh', 40 )
    max_val = kwargs.get( 'max_val', 100 )
    max_flow_val = kwargs.get( 'max_flow_val', 0 )

    max_flow_val = min( max_flow_val, max_val )
    red = max( 0, max_flow_val - gr_thresh ) / ( max_val - gr_thresh )
    assert red >= 0.0 and red <= 1.0
    green = 1 - red

    red_int = int( 255 * red )
    green_int = int( 255 * green )

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, green_int, red_int))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, green_int, red_int), -1)
    return vis


if __name__ == '__main__':

    video_name1 = 'bird.avi'
    video_name1 = 'brazil.mp4'
    video_name1 = 'new-orleans.mp4'
    video_name1 = 'new-orleans-short-edited.mp4'
    video_name = 'IMG_3150.MOV'
    
    cam = cv2.VideoCapture( video_name )
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.resize(prevgray, (int(0.4*prevgray.shape[1]), int(0.4*prevgray.shape[0])))
    show_hsv = False

    height, width, layers = prev.shape
	
    root = video_name.split( '.' )[0] + '_'
    out_flow = cv2.VideoWriter(root + "vector_flow.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,(width,height))
    out_hsv = cv2.VideoWriter(root + "hsv_flow.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,(width,height))

    ## while True:
    window = []
    window_size = 10
    for i in range( 100000 ):
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (int(0.4*gray.shape[1]), int(0.4*gray.shape[0])))
        flow = 10*cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        max_flow_val = 0
        if i > 5:
            window.insert( 0, flow.max() )
            if len( window ) > window_size:
                window.pop()
            max_flow_val = max( window )

        prevgray = gray

        cur_flow = my_draw_flow(gray, flow, max_flow_val= max_flow_val)
        #cur_hsv = of.draw_hsv(flow)
		
        out_flow.write( cur_flow )
        #out_hsv.write( cur_hsv )
		
        cv2.imshow('flow', cur_flow)
		
        #if show_hsv:
            #cv2.imshow('flow HSV', cur_hsv)

        ch = cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
    cv2.destroyAllWindows()
    out_flow.release()
    out_hsv.release()


# In[ ]:





# In[ ]:




