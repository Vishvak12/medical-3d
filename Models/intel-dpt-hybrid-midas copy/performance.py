from math import log10, sqrt
import cv2
import numpy as np
import numpy

from PIL import Image,ImageChops
import math, operator
from skimage.metrics import structural_similarity as ssim
from skimage import color 




  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def MSE(imageA, imageB):
	
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err/1000

# def rmsdiff(im1, im2):
#     "Calculate the root-mean-square difference between two images"
#     diff = ImageChops.difference(im1, im2)
#     h = diff.histogram()
#     sq = (value*((idx%256)**2) for idx, value in enumerate(h))
#     sum_of_squares = sum(sq)
#     rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
#     return rms
def SSIM(imageA, imageB):
 # Calculate the MSE and SSIM
 m = MSE(imageA, imageB)
 s = ssim(imageA, imageB)

 # Return the SSIM. The higher the value, the more "similar" the two images are.
 return s


def RMSE(actual,predicted):

  MSE = np.square(np.subtract(actual,predicted)).mean()   
  rmse = math.sqrt(MSE) 
  return rmse/100

def standard_deviation(pre_img):
    SD=pre_img.stddev
    return SD
def Average_Gradient(pre_img):
    lap = cv2.Laplacian(pre_img,cv2.CV_64F,ksize=3) 
    lap = np.uint8(np.absolute(lap))
    return lap
def corrcoef(a,b):
    a = a - numpy.mean(a)
    b = b - numpy.mean(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r 

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def calc_nmse(_img_ref, _img_eva):
    # Calc NMSE
    numer = np.sum((_img_ref - _img_eva) ** 2)
    denom = np.sum(_img_ref ** 2)

    print(f'numer: {numer}')
    print(f'denom: {denom}')

    nmse = numer / denom
    return nmse


def calc_statistics( _img ):
    print( 'statistics:' )
    print( 'max   : {}'.format( np.max(_img) ) )
    print( 'min   : {}'.format( np.min(_img) ) )
    print( 'mean  : {:.0f} ({:.0f})'.format( np.mean(_img), _img[_img != 0].mean() ) )
    print( 'std   : {:.0f} ({:.0f})'.format( np.std(_img), _img[_img != 0].std() ) )


def calc_entropy( _histogram, _num_of_pixels ):
    entropy = 0

    for i in range( 256 ):
        # Calc the probability of appearance of pixel value "i"
        p = _histogram[i] / _num_of_pixels

        if p == 0:
            continue
        entropy -= p * np.log2( p )

    return entropy

#average difference
def avgdiff(imgA,imgB):
    rows,cols=imgA.shape
    sumimg=0
    for i in range(rows):
        for j in range(cols):
            sumimg=sumimg+(float(imgA[i,j])-float(imgB[i,j]))
    return sumimg/(rows*cols)

#Normalized correlation
def normalizedcorrelation(imgA,imgB):
    rows,cols=imgA.shape
    sqrAsum=0
    productsum=0
    for i in range(rows):
        for j in range(cols):
            productsum=productsum+(float(imgA[i,j])*float(imgB[i,j]))
            sqrAsum=sqrAsum+((float(imgA[i,j]))**2)
   
    return productsum/sqrAsum
    

#Maximum Difference

def maxdiff(imgA,imgB):
    diff=np.absolute(imgA.astype('float')-imgB.astype('float'))
    return np.amax(diff)

#Normalized Absolute error
def normalizedabsoluterr(imgA,imgB):
    rows,cols=imgA.shape
    Asum=0
    productsum=0
    for i in range(rows):
        for j in range(cols):
            product=abs(float(imgA[i,j])-float(imgB[i,j]))
            productsum=productsum+product
            Asum=Asum+abs(float(imgA[i,j]))
    return productsum/(Asum)

#Structural Content
def structuralcontent(imgA,imgB):
    rows,cols=imgA.shape
    Asum=0
    Bsum=0
    for i in range(rows):
        for j in range(cols):
            Asum=Asum+(float(imgA[i,j]))**2
            Bsum=Bsum+(float(imgB[i,j]))**2
   
    return Asum/Bsum
#SSIM Structural SIMilarity Index (SSIM).
def ssim(imgA,imgB):
    def __get_kernels():
        k1, k2, l = (0.01, 0.03, 255.0)
        kern1, kern2 = map(lambda x: (x * l) ** 2, (k1, k2))
        return kern1, kern2