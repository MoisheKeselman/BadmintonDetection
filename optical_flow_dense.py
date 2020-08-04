import numpy as np
import cv2 as cv
#cap = cv.VideoCapture(cv.samples.findFile("vtest.avi"))
cap = cv.VideoCapture("IMG_3150.MOV")
ret, frame1 = cap.read()
frame1_small = cv.resize(frame1, (int(0.4*frame1.shape[1]), int(0.4*frame1.shape[0])))
prvs = cv.cvtColor(frame1_small,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1_small)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    next_small = cv.resize(next, (int(0.4*next.shape[1]), int(0.4*next.shape[0])))
    flow = cv.calcOpticalFlowFarneback(prvs,next_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next_small
