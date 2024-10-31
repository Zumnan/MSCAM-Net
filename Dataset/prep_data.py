import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define the dataset directory
DATASET_DIR = "ISIC"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")


PROCESSED_DIR = "preprocessed_data1"
TRAIN_IMAGES_DIR = os.path.join(PROCESSED_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(PROCESSED_DIR, "train", "masks")
VALID_IMAGES_DIR = os.path.join(PROCESSED_DIR, "valid", "images")
VALID_MASKS_DIR = os.path.join(PROCESSED_DIR, "valid", "masks")

os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_MASKS_DIR, exist_ok=True)
os.makedirs(VALID_IMAGES_DIR, exist_ok=True)
os.makedirs(VALID_MASKS_DIR, exist_ok=True)


def load_image(image_path):
    image = cv2.imread(image_path)
    return image


def resize_image(image, target_resolution=(256, 256)):
    
    resized_image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_AREA)
    return resized_image


def create_and_write_image_mask(image_paths, mask_paths, save_images_dir, save_masks_dir, target_resolution=(512, 512)):
    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Processing images and masks"):
        
        image = load_image(image_path)
        mask = load_image(mask_path)

        
        resized_image = resize_image(image, target_resolution)
        resized_mask = resize_image(mask, target_resolution)

        
        image_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        mask_filename = os.path.basename(mask_path)  # already in PNG format

        cv2.imwrite(os.path.join(save_images_dir, image_filename), resized_image)
        cv2.imwrite(os.path.join(save_masks_dir, mask_filename), resized_mask)


image_files = sorted([os.path.join(IMAGES_DIR, filename) for filename in os.listdir(IMAGES_DIR) if filename.endswith(".jpg")])
mask_files = sorted([os.path.join(MASKS_DIR, filename) for filename in os.listdir(MASKS_DIR) if filename.endswith(".png")])


train_images, valid_images, train_masks, valid_masks = train_test_split(image_files, mask_files, train_size=0.8, shuffle=True, random_state=42)


create_and_write_image_mask(train_images, train_masks, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, target_resolution=(256, 256))
create_and_write_image_mask(valid_images, valid_masks, VALID_IMAGES_DIR, VALID_MASKS_DIR, target_resolution=(256, 256))
