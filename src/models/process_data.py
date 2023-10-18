# importing libraries 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from os import listdir
from os.path import isfile, join
from src import METRICS_DIR, PROCESSED_DATA_DIR,RAW_DATA_DIR
from pathlib import Path

def noise_removal(image_path:Path):
    
    # Reading image from folder where it is stored 
    img = cv2.imread(image_path) 

    # denoising of image saving it into dst image 
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
    cv2.imwrite(image_path,denoised)
    return denoised
