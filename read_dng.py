#!/usr/bin/env python

import rawpy
import cv2

#raw = rawpy.imread('data/20180912_223321.dng')
raw = rawpy.imread('data/10.dng')
rgb = raw.postprocess()

#visualization
cv2.namedWindow('raw',cv2.WINDOW_NORMAL)
cv2.resizeWindow('raw', 1600,1600)
cv2.imshow('raw', rgb[:,:,[2,1,0]])
cv2.waitKey(0)

# store on file
cv2.imwrite('a.tiff', rgb[:,:,[2,1,0]] )


