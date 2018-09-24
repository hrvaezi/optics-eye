#!/usr/bin/env python

import numpy as np
import cv2
import sys
import glob


def frame_list(path):
  folders = ['g1', 'g2', 'g3'] 
  fns = []
  for f in folders:
    f1 = glob.glob(path + '/' + f + '/*.tif')
    f1.sort()
    fns.extend(f1[:-2]) 
   
  return fns

pnts = []

def click_and_crop(event, x, y, flags, param):
  global pats
  if event == cv2.EVENT_LBUTTONDOWN:
    pnts.append((x,y))
    print(x,y)

# Create some random colors
color = np.random.randint(0,255,(100,3))

def get_pts(fn):
  global pnts
  cv2.namedWindow("image",cv2.WINDOW_NORMAL)
  cv2.resizeWindow("image", 1600,1600)
  cv2.setMouseCallback("image", click_and_crop)
  img = ((cv2.imread(fn, -1) / 4 )).astype(np.uint8)
  img2 = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
  img2[:,:,0] = img
  img2[:,:,1] = img
  img2[:,:,2] = img
  print(img2.shape,img.max(), img2.min(),img2.mean())
  while True:
    # display the image and wait for a keypress
    for i in range(len(pnts)):
       img2 = cv2.circle(img2,(pnts[i][0],pnts[i][1]),10,color[i].tolist(),-1)
    cv2.imshow("image", img2)
    key = cv2.waitKey(1) & 0xFF
 
    if key == 27:
      break
  
  return pnts
 


def track(fns, pats):
  # params for ShiTomasi corner detection
  feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
  # Parameters for lucas kanade optical flow
  lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

  cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
  cv2.resizeWindow("frame", 1600,1600)
  
  # Take first frame and find corners in it
  old_gray = ((cv2.imread(fns[0], -1) / 4 )).astype(np.uint8)
  #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
  p0 = np.expand_dims(np.array(pnts).astype(np.float32), axis=1) 
  print(p0.shape, len(pnts), p0[1])
  vid = []
  # Create a mask image for drawing purposes
  c_size = [old_gray.shape[0], old_gray.shape[1], 3]
  mask = np.zeros(c_size, dtype=np.uint8)
  frame = np.zeros(c_size, dtype=np.uint8) 
  for i in range(1, len(fns)):
    print(fns[i])
    frame_gray = ((cv2.imread(fns[i], -1) / 4 )).astype(np.uint8)
    frame[:,:,0] = frame[:,:,1] = frame[:,:,2] = frame_gray 
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),15,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    vid.append(img)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
  cv2.destroyAllWindows()
  
  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('output.avi',fourcc, 15.0, (2048, 2048))

  for v in vid:
    # write the flipped frame
    out.write(v)
  out.release()

fns = frame_list(sys.argv[1]) 
get_pts(fns[0])
print(fns)
track(fns, pnts)



exit(0)




cap = cv2.VideoCapture('slow.flv')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()