"""Module process data"""
# importing libraries
import os
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2
import numpy as np

from src import PROCESSED_DATA_DIR, RAW_DATA_DIR


def noise_removal(img_path):
    """Function for noise removal."""
    # Reading image from folder where it is stored
    img = cv2.imread(img_path)
    # denoising of image saving it into dst image
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    # cv2.imwrite(dst_path,denoised)
    return denoised


def rgb2gray(img_path):
    """Function rgb2gray."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def get_images_path(directory_name):
    """Function that gets images path."""
    imgs_path={}
    for file in listdir(directory_name):
        if isfile(join(directory_name, file)):
            imgs_path[file] = join(directory_name /file)
    return imgs_path

def auto_canny_edge_detection(path, sigma=0.33):
    """
    Perform automatic Canny edge detection on an image.
    This function reads an image from the given file path, then applies the Canny edge
    detection algorithm with threshold values determined automatically. The thresholds are
    set based on the median pixel intensity of the image, adjusted by a factor defined by
    the 'sigma' parameter.
    """
    image = cv2.imread(path)
    median = np.median(image)
    lower_value = int(max(0, (1.0-sigma) * median))
    upper_value = int(min(255, (1.0+sigma) * median))
    return cv2.Canny(image, lower_value, upper_value)

def image_processing(src_imgs_dir,dst_imgs_dir):
    """Function that processes images"""
    imgs_path=get_images_path(src_imgs_dir)

    for image_name,path in imgs_path.items():
        denoised=noise_removal(path)
        gray=cv2.cvtColor(denoised,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(dst_imgs_dir / image_name),gray)

def copy_labels(src,dest):
    """
    Copy all files from a source directory to a destination directory.
    This function lists all files in the source directory specified by 'src',
    and then copies each file to the destination directory specified by 'dest'.
    It does not copy subdirectories or their contents, only files directly
    within the source directory.
    """
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)

def main():
    """
    Main function
    """
    train_imgs_dir=RAW_DATA_DIR / "train/images"
    valid_imgs_dir=RAW_DATA_DIR / "valid/images"
    test_imgs_dir=RAW_DATA_DIR / "test/images"
    processed_train_imgs_dir=PROCESSED_DATA_DIR / "train/images"
    processed_test_imgs_dir=PROCESSED_DATA_DIR / "test/images"
    processed_valid_imgs_dir=PROCESSED_DATA_DIR / "valid/images"

    train_labels=PROCESSED_DATA_DIR / "train/labels"
    test_labels=PROCESSED_DATA_DIR / "test/labels"
    valid_labels=PROCESSED_DATA_DIR / "valid/labels"

    Path(processed_train_imgs_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_test_imgs_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_valid_imgs_dir).mkdir(parents=True, exist_ok=True)

    Path(train_labels).mkdir(parents=True, exist_ok=True)
    Path(test_labels).mkdir(parents=True, exist_ok=True)
    Path(valid_labels).mkdir(parents=True, exist_ok=True)

    copy_labels(RAW_DATA_DIR / "train/labels",train_labels)
    copy_labels(RAW_DATA_DIR / "test/labels",test_labels)
    copy_labels(RAW_DATA_DIR / "valid/labels",valid_labels)

    image_processing(train_imgs_dir,processed_train_imgs_dir)
    image_processing(test_imgs_dir,processed_test_imgs_dir)
    image_processing(valid_imgs_dir,processed_valid_imgs_dir)

if __name__ == "__main__":
    main()
