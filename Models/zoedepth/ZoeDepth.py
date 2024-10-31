import os
import cv2
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
from image_enhancement import image_enhancement
from performance import (
    PSNR, MSE, SSIM, RMSE, standard_deviation, Average_Gradient, corrcoef,
    signaltonoise, calc_nmse, avgdiff, normalizedcorrelation, maxdiff,
    normalizedabsoluterr, structuralcontent
)
import open3d as o3d
from D3FFT import D3FFT
# Configuration
IMAGE_SIZE = (256, 256)
METHODS = ['ABF', 'BF', 'GF', 'AMF', 'MF']
RESULT_DIRS = {
    "Original": "Original_img/",
    "ABF": "ABF_img/",
    "BBHE": "BBHE_img/",
    "FeatureDetect": "FeatureDetect_img/",
    "FeatureMatching": "FeatureMatching_img/",
    "Register": "Register_img/",
    "Register2D": "Register_2D_image/"
}

# Ensure directories exist
for dir_path in RESULT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Helper functions
def load_images_from_folder(folder_path):
    images = []
    for r, d, f in os.walk(folder_path):
        for file in f:
            images.append(os.path.join(r, file))
    return images

def preprocess_images(image_path, size=IMAGE_SIZE):
    """Load, resize, and convert to grayscale."""
    image_data = cv2.imread(image_path)
    resized_img = cv2.resize(image_data, size)
    gray_img = color.rgb2gray(resized_img)
    return resized_img, gray_img

def bilateral_filter(image, diameter=15, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def save_image(image, filename):
    cv2.imwrite(filename, image)

def image_enhancement_bbhe(image, color_space='HSV'):
    ie = image_enhancement.IE(image, color_space=color_space)
    return ie.BBHE()

# Load and preprocess images
images_ct1 = load_images_from_folder('right_test//')
images_ct2 = load_images_from_folder('left_test//')

# Processing pipeline
for i, (image_ct1_path, image_ct2_path) in enumerate(zip(images_ct1, images_ct2)):
    # Load and preprocess images
    Right_resize_img, Right_resize_gray_img = preprocess_images(image_ct1_path)
    Left_resize_img, Left_resize_gray_img = preprocess_images(image_ct2_path)
    
    # Save original images
    save_image(Right_resize_img, os.path.join(RESULT_DIRS["Original"], f"org_R_{i}.png"))
    save_image(Left_resize_img, os.path.join(RESULT_DIRS["Original"], f"org_L_{i}.png"))

    # Apply bilateral filter
    Right_bilateral = bilateral_filter(Right_resize_img)
    Left_bilateral = bilateral_filter(Left_resize_img)

    # Save filtered images
    save_image(Right_bilateral, os.path.join(RESULT_DIRS["ABF"], f"ABF_R_{i}.png"))
    save_image(Left_bilateral, os.path.join(RESULT_DIRS["ABF"], f"ABF_L_{i}.png"))

    # BBHE Enhancement
    Right_BBHE_img = image_enhancement_bbhe(Right_bilateral)
    Left_BBHE_img = image_enhancement_bbhe(Left_bilateral)
    save_image(Right_BBHE_img, os.path.join(RESULT_DIRS["BBHE"], f"BBHE_R_{i}.png"))
    save_image(Left_BBHE_img, os.path.join(RESULT_DIRS["BBHE"], f"BBHE_L_{i}.png"))

    # Feature detection and matching
    dst1 = cv2.cornerHarris(np.float32(color.rgb2gray(Right_BBHE_img)), 2, 3, 0.04)
    Right_BBHE_img[dst1 > 0.01 * dst1.max()] = [0, 255, 0]
    save_image(Right_BBHE_img, os.path.join(RESULT_DIRS["FeatureDetect"], f"Fea_det_R_{i}.png"))

    dst2 = cv2.cornerHarris(np.float32(color.rgb2gray(Left_BBHE_img)), 2, 3, 0.04)
    Left_BBHE_img[dst2 > 0.01 * dst2.max()] = [0, 255, 0]
    save_image(Left_BBHE_img, os.path.join(RESULT_DIRS["FeatureDetect"], f"Fea_det_L_{i}.png"))

    # Assuming D3FFT function is properly defined elsewhere in your code
    fea_matching_img, Register_img, Register_gray_img = D3FFT(Right_BBHE_img, Left_BBHE_img, Right_resize_img, Left_resize_img)
    save_image(fea_matching_img, os.path.join(RESULT_DIRS["FeatureMatching"], f"Fea_mat_R_{i}.png"))
    save_image(Register_img, os.path.join(RESULT_DIRS["Register"], f"{i}_1.png"))

# Depth estimation with ZoeDepth model
feature_extractor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")

def depth_estimation(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()
    return predicted_depth

# Apply depth estimation on registered images
for j, register_image in enumerate(load_images_from_folder(RESULT_DIRS["Register2D"])):
    output = depth_estimation(register_image)
    plt.imshow(output, cmap='plasma')
    plt.show()

# Visualize results
def plot_performance_metrics(metrics, labels, ylabel, title):
    plt.bar(labels, metrics, color='cadetblue')
    plt.ylabel(ylabel, fontsize=16, fontname='Times New Roman')
    plt.title(title, fontname='Times New Roman', fontsize=14)
    plt.show()

# Example metric plot
plot_performance_metrics(
    metrics=[37.86, 33.2, 31.2, 28.4, 24.6],
    labels=METHODS,
    ylabel='Peak signal-to-noise ratio',
    title='Noise Reduction Performance'
)

# Define additional metrics and plots as necessary
# End of code
