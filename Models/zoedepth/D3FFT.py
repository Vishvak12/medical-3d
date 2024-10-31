    
import numpy
import cv2
from skimage import color 


def D3FFT(Right_BBHE_img,Left_BBHE_img,Right_resize_img,Left_resize_img):
  orb = cv2.ORB_create()
  queryKeypoints, queryDescriptors = orb.detectAndCompute(Right_BBHE_img,None)
  trainKeypoints, trainDescriptors = orb.detectAndCompute(Left_BBHE_img,None)
  matcher = cv2.BFMatcher()
  matches = matcher.match(queryDescriptors,trainDescriptors)
  fea_matching_img = cv2.drawMatches(Right_BBHE_img, queryKeypoints,
  Left_BBHE_img, trainKeypoints, matches[:20],None)
  Register_img=Right_resize_img+Left_resize_img
  Register_gray_img = color.rgb2gray(Register_img)
  return fea_matching_img,Register_img,Register_gray_img