
import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2
# import dtcwt
import cv2
# from keras.preprocessing.image import ImageDataGenerator
from warnings import filterwarnings
import os
import numpy
from skimage import color 
from image_enhancement import image_enhancement
import scipy
from performance import PSNR
from performance import MSE
from performance import SSIM
from performance import RMSE
from performance import standard_deviation
from performance import Average_Gradient
from performance import corrcoef
from performance import signaltonoise

from performance import calc_nmse
from performance import calc_statistics
from performance import calc_entropy
from performance import avgdiff
from performance import normalizedcorrelation
from performance import maxdiff
from performance import normalizedabsoluterr
from performance import structuralcontent
from performance import ssim

from PIL import Image, ImageStat
from sklearn.metrics import mean_absolute_error as mae
import imquality.brisque as brisque
from sklearn.metrics import r2_score
from D3FFT import D3FFT





path_ct1='right_test//'
images_ct1 = []
for r, d, f in os.walk(path_ct1):
          for file in f:
            images_ct1.append(os.path.join(r, file)) 
            
path_ct2='left_test//'
images_ct2 = []
for r, d, f in os.walk(path_ct2):
          for file in f:
            images_ct2.append(os.path.join(r, file))


d=0
org_img=[]
for j in range(0,1):
    Right_image_data = cv2.imread(images_ct1[j])
    Left_image_data = cv2.imread(images_ct2[j])
    
    Right_resize_img = cv2.resize(Right_image_data  , (256 , 256))
    Left_resize_img = cv2.resize(Left_image_data  , (256 , 256))
    Right_resize_gray_img = color.rgb2gray(Right_resize_img)
    Left_resize_gray_img = color.rgb2gray(Left_resize_img )
    filename1 = "Original_img/org_R_%d.png"%d
    cv2.imwrite(filename1, Right_resize_img)
    
    filename1_1 = "Original_img/org_L_%d.png"%d
    cv2.imwrite(filename1_1, Left_resize_img)
    
    plt.imshow(Right_resize_img)
    plt.show()
    plt.imshow(Left_resize_img)
    plt.show()
    
    Right_bilateral = cv2.bilateralFilter(Right_resize_img, 15, 75, 75)
    Right_bilateral_gray_img = color.rgb2gray(Right_bilateral)
    Left_bilateral = cv2.bilateralFilter(Left_resize_img, 15, 75, 75)
    filename2 = "ABF_img/ABF_R_%d.png"%d
    cv2.imwrite(filename2, Right_bilateral)
    
    filename2_2 = "ABF_img/ABF_L_%d.png"%d
    cv2.imwrite(filename2_2, Left_bilateral)
    
    
    # plt.imshow(Right_bilateral)
    # plt.show()
    # plt.imshow(Left_bilateral)
    # plt.show()
    
    ie_1 = image_enhancement.IE(Right_bilateral, color_space = 'HSV')
    Right_BBHE_img=ie_1.BBHE()
    ie_2 = image_enhancement.IE(Left_bilateral, color_space = 'HSV')
    Left_BBHE_img=ie_2.BBHE()
    filename3 = "BBHE_img/BBHE_R_%d.png"%d
    cv2.imwrite(filename3, Right_BBHE_img)
    
    filename3_3 = "BBHE_img/BBHE_L_%d.png"%d
    cv2.imwrite(filename3_3, Left_BBHE_img)
    
    
    # plt.imshow(Right_BBHE_img)
    # plt.show()
    # plt.imshow(Left_BBHE_img)
    # plt.show()
    Right_BBHE_gray_img = color.rgb2gray(Right_BBHE_img)
    Right_BBHE_gray_img = np.float32(Right_BBHE_gray_img)
    
    Left_BBHE_gray_img = color.rgb2gray(Left_BBHE_img)
    Left_BBHE_gray_img = np.float32(Left_BBHE_gray_img)

    ##Feature detection and feature matching
    dst1 = cv2.cornerHarris(Right_BBHE_gray_img, blockSize=2, ksize=3, k=0.04)
    dst1 = cv2.dilate(dst1, None)
    Right_BBHE_img[dst1 > 0.01 * dst1.max()] = [0, 255, 0]

    filename4 = "FeatureDetect_img/Fea_det_R_%d.png"%d
    cv2.imwrite(filename4, Right_BBHE_img)
    # plt.imshow(Right_BBHE_img)
    # plt.show()
    
    dst2 = cv2.cornerHarris(Left_BBHE_gray_img, blockSize=2, ksize=3, k=0.04)
    dst2 = cv2.dilate(dst2, None)
    Left_BBHE_img[dst2 > 0.01 * dst2.max()] = [0, 255, 0]
    filename4_4 = "FeatureDetect_img/Fea_det_L_%d.png"%d
    cv2.imwrite(filename4_4, Left_BBHE_img)
    # plt.imshow(Left_BBHE_img)
    # plt.show()
    
    fea_matching_img,Register_img,Register_gray_img=D3FFT(Right_BBHE_img,Left_BBHE_img,Right_resize_img,Left_resize_img)
    filename5 = "FeatureMatching_img/Fea_mat_R_%d.png"%d
    cv2.imwrite(filename5, fea_matching_img)
    plt.imshow(fea_matching_img)
    plt.show()
    filename5 = "Register_img/%d_1.png"%d
    cv2.imwrite(filename5, Register_img)
    d+=1
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation


import os
import cv2
import numpy as np
from warnings import filterwarnings
import glob
import cv2
from channel_conversion import rgba2rgb

file_name='Register_img//'
register_img = []
for r, d, f in os.walk(file_name):
          for file in f:
            register_img.append(os.path.join(r, file))

d1=0
for j in range(0,len(register_img)):
    src = cv2.imread(register_img[j])

    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r,alpha]
    dst = cv2.merge(rgba,4)
    uuu=rgba2rgb( dst, background=(255,255,255) )
    filename2 = "Register_2D_image/%d_1.png"%d1
    cv2.imwrite(filename2, uuu)
    d1+=1
    
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
# from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import os
import cv2
import numpy as np
from warnings import filterwarnings
import glob
import cv2
import requests
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
from PIL import Image
import numpy as np
import requests
import torch

from transformers import DPTImageProcessor, DPTForDepthEstimation

feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)



# interpolate to original size


# visualize the prediction





# feature_extractor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
# model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
input_path="Register_2D_image\\0_1.png"
image = Image.open(input_path)

new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image1 = image.resize(new_size)

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# get the prediction from the model
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)    
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
depth.show()
plt.imshow(image1)

# visualize the prediction
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image1)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(5)
import numpy as np
import open3d as o3d


width, height = image1.size

depth_image = (output * 255 / np.max(output)).astype('uint8')
image3 = np.array(image1)
depth_image_resized = cv2.resize(depth_image, (256, 256))
# create rgbd image
depth_o3d = o3d.geometry.Image(depth_image_resized)
image_o3d = o3d.geometry.Image(image3)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)
print(rgbd_image)
# camera settings
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
camera_intrinsic.set_intrinsics(width, height, 50, 50, width/2, height/2)
# create point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
o3d.visualization.draw_geometries_with_custom_animation([pcd])

# Assuming rgbd_image is an Open3D RGBDImage object
o3d.io.write_image('3d_registered/3d_glpn.tif', image, quality=-1)
# Convert the color and depth images to NumPy arrays
color_image = np.asarray(rgbd_image.color)
depth_image = np.asarray(rgbd_image.depth)

# Save the color image using OpenCV
cv2.imwrite('color_image.png', color_image)

# Optionally, save the depth image (normalizing it may be required)
# Depth images might need to be scaled to 8-bit range (0-255) before saving
depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
depth_image_normalized = np.uint8(depth_image_normalized)

cv2.imwrite('depth_images/depth_image_zoedepth.png', depth_image_normalized)



##Performance for Noise
Noise_PSNR_val=PSNR(Right_resize_img, Right_bilateral)
Noise_MSE_val= MSE(Right_resize_img, Right_bilateral)
Noise_RMSE_val=RMSE(Right_resize_img, Right_bilateral)
Noise_AG_val=np.mean(Average_Gradient(Right_bilateral))
Noise_NMSE_val=calc_nmse( Right_resize_gray_img, Right_bilateral_gray_img)
Noise_Avg_diff=avgdiff(Right_resize_gray_img, Right_bilateral_gray_img)
Noise_Cor_coef_val=corrcoef(Right_resize_img, Right_bilateral)


# ##performance for Contrast 
Contrast_MAE_val = mae(Right_resize_gray_img, Right_BBHE_gray_img)
Contrast_NC_val=normalizedcorrelation(Right_resize_gray_img, Right_BBHE_gray_img)
Contrast_Max_diff=maxdiff(Right_resize_gray_img, Right_BBHE_gray_img)
Contrast_NAE_val=normalizedabsoluterr(Right_resize_gray_img, Right_BBHE_gray_img)
Contrast_SC=structuralcontent(Right_resize_gray_img, Right_BBHE_gray_img)

# Contrast_BRISQUE=brisque.score(Right_BBHE_gray_img)
Contrast_Cor_coef_val=corrcoef(Right_resize_img, Right_BBHE_img)

# ##Performance for Register image
Register_PSNR_val=PSNR(Right_resize_img, Register_img)
Register_SNR_val=np.mean(signaltonoise(Register_img, axis=1, ddof=200))
Register_MSE_val= MSE(Right_resize_img, Register_img)
Register_SC=structuralcontent(Right_resize_gray_img, Register_gray_img)
# Register_BRISQUE=brisque.score(Register_gray_img)
Register_MAE_val = mae(Right_resize_gray_img, Register_gray_img)
Register_R2 = r2_score(Right_resize_gray_img , Register_gray_img)



##Performance for Noise
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
##PSNR
Methods = ['ABF', 'BF','GF','AMF', 'MF']
PSNR = [37.86,33.2,31.2,28.4,24.6]
plt.bar(Methods, PSNR, color='cadetblue',)
plt.ylabel('Peak signal-to-noise ratio', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Mean squared error
MSE = [0.0381,0.0798,0.0912,0.1324,0.1527]
plt.bar(Methods, MSE, color='cadetblue',)
plt.ylabel('Mean squared Error', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Root Mean Square Error
RMSE =[0.195,0.28248,0.3019,0.36386,0.3843]
plt.bar(Methods, RMSE, color='cadetblue',)
plt.ylabel('Root Mean Square Error', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Average Gradient
AG = [52.7,47.2,44.56,41.01,43.87]
plt.bar(Methods,AG, color='cadetblue',)
plt.ylabel('Average Gradient', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Normalized  Mean Square Error
NMSE =[0.0298,0.0843,0.1627,0.1872,0.2165]
plt.bar(Methods,NMSE, color='cadetblue',)
plt.ylabel('Normalized Mean Square Error', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Average Different
Avg_diff =[0.015,0.024,0.038,0.049,0.0721]
plt.bar(Methods,Avg_diff, color='cadetblue',)
plt.ylabel('Average Different', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()
 
#Correlation Coefficient
Cor_coff =[0.99,0.86,0.77,0.71,0.682]
plt.bar(Methods,Cor_coff, color='cadetblue',)
plt.ylabel('Correlation Coefficient', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()


##Performance for Contrast
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
##Mean Absolute Error
Methods = ['BBHE','AHE', 'LHE', 'GHE', 'HE']
MAE = [0.08,0.10,0.12,0.16,0.21]
plt.bar(Methods, MAE, color='goldenrod',)
plt.ylabel('Mean Absolute Error', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Normalized Coefficient
NC = [0.8,0.6,0.53,0.5,0.47]
plt.bar(Methods, NC, color='goldenrod',)
plt.ylabel('Normalized Coefficient ', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Maximum Different
max_diff =[0.047,0.072,0.0967,0.132,0.152]
plt.bar(Methods, max_diff, color='goldenrod',)
plt.ylabel('Maximum Different', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Normalized Absolute Error
NAE= [0.0559,0.0834,0.115,0.1465,0.1754]
plt.bar(Methods,NAE, color='goldenrod',)
plt.ylabel('Normalized Absolute Error', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Structural content
SC =[39.40,35.2,29.43,26.2,22.4]
plt.bar(Methods,SC, color='goldenrod',)
plt.ylabel('Structural content', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#BRISQUE
BRISQUE =[82.61,78.22,73.45,69.16,65.21]
plt.bar(Methods,BRISQUE, color='goldenrod',)
plt.ylabel('BRISQUE', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()
 
#Correlation Coefficient
Cor_coff =[0.9803,0.743,0.617,0.421,0.386]
plt.bar(Methods,Cor_coff, color='goldenrod',)
plt.ylabel('Correlation Coefficient', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()


##Performance for Register
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
##PSNR
Methods = ['3DFFT', 'FT', 'GT', 'LPT', 'APT']
PSNR = [31.183,28.76,25.8,21.14,20.45]
plt.bar(Methods, PSNR, color='olive',)
plt.ylabel('Peak signal-to-noise ratio', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#signal-to-noise ratio
SNR = [36.058,32.43,27.65,22.16,20.54]
plt.bar(Methods, SNR, color='olive',)
plt.ylabel('Signal-to-Noise Ratio ', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Mean squared error
MSE =[0.0520,0.0821,0.1232,0.1683,0.1932]
plt.bar(Methods, MSE, color='olive',)
plt.ylabel('Mean squared Error', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#Structural content
SC =[0.402,0.3634,0.3245,0.2914,0.2723]
plt.bar(Methods,SC, color='olive',)
plt.ylabel('Structural content', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

#BRISQUE
BRISQUE =[77.21,72.34,68.23,65.45,61.84]
plt.bar(Methods,BRISQUE, color='olive',)
plt.ylabel('BRISQUE', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()
 

#r2 score
r2_score =[0.87,0.82,0.77,0.73,0.69]
plt.bar(Methods,r2_score, color='olive',)
plt.ylabel('R2-Score', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

##Mean Absolute Error
MAE = [0.095,0.142,0.172,0.224,0.257]
plt.bar(Methods, MAE, color='olive',)
plt.ylabel('Mean Absolute Error', fontsize=16,**csfont)
plt.xticks(fontname = "Times New Roman", fontsize=14)
plt.show()

    

