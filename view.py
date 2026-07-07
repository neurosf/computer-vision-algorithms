import os
import sys
import pickle
import cv2
import numpy as np
import glob
import logging
import re


class View:
    """Represents an image used in the reconstruction"""

    def __init__(self, image_path, root_path):
        self.name = image_path[image_path.rfind('/') + 1:-4]  # image name without extension
        self.image = cv2.imread(image_path)  # numpy array of the image
        self.keypoints = []  # list of keypoints obtained from feature extraction
        self.descriptors = []  # list of descriptors obtained from feature extraction
        self.root_path = root_path  # root directory containing the image folder
        self.R = np.zeros((3, 3), dtype=float)  # rotation matrix for the view
        self.t = np.zeros((3, 1), dtype=float)  # translation vector for the view

        self.extract_features()


    def extract_features(self):
        """Extracts features from the image"""
        detector = cv2.SIFT_create()
        self.keypoints, self.descriptors = detector.detectAndCompute(self.image, None)
        logging.info("Computed features for image %s", self.name)

        self.draw_keypoints()  # Draw keypoints after extracting features

    def draw_keypoints(self):
        """Draws keypoints on the image and saves the result"""
        image_with_keypoints = cv2.drawKeypoints(self.image, self.keypoints, None, color=(0, 255, 0),
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        output_path = os.path.join(self.root_path, 'keypoints', os.path.basename(self.name) + '_keypoints.jpg')

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        cv2.imwrite(output_path, image_with_keypoints)
        logging.info("Keypoints drawn and saved for image %s", self.name)

def numerical_sort(value):
    """Sort function to extract numerical values from strings."""
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else float('inf')

def create_views(root_path, image_format='jpg'):
    """Loops through the images and creates an array of views"""

    image_names = sorted(glob.glob(os.path.join(root_path, 'images', '*.' + image_format)), key=numerical_sort)

    logging.info("Computing features")
    views = []
    for image_name in image_names:
        views.append(View(image_name, root_path))

    return views