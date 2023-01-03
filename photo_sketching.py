"""
@author: tenkapo
"""

import cv2 as cv


#===================== load image and convert to grayscale  ==============================
img = cv.imread('./om_squad_2020.jpg')  #read the image

# Display original image
cv.imshow('Original', img)
cv.waitKey(0)
 
# Convert to graycsale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', img_gray)
cv.waitKey(0)


#===================== invert image and blur the image  ==============================
img_invert = cv.bitwise_not(img_gray) #inversion
cv.imshow('invert', img_invert)
cv.waitKey(0)

img_blur = cv.GaussianBlur(img_invert, (5,5), 0) #blur
cv.imshow('smooth', img_blur)
cv.waitKey(0)

#===================== sketch  ==============================
img_sketch = cv.divide(img_gray, 255 - img_blur, scale=255)
cv.imshow('sketch', img_sketch)
cv.waitKey(0)