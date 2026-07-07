import os
import pickle
import cv2
import logging
import numpy as np

class Match:
    """Represents feature matches between two views"""

    def __init__(self, view1, view2):
        self.indices1 = []  # indices of the matched keypoints in the first view
        self.indices2 = []  # indices of the matched keypoints in the second view
        self.distances = []  # distance between the matched keypoints in the first view
        self.image_name1 = view1.name  # name of the first view
        self.image_name2 = view2.name  # name of the second view
        self.root_path = view1.root_path  # root directory containing the image folder
        self.inliers1 = []  # list to store the indices of the keypoints from the first view not removed using the fundamental matrix
        self.inliers2 = []  # list to store the indices of the keypoints from the second view not removed using the fundamental matrix
        self.view1 = view1
        self.view2 = view2

        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        self.get_matches(view1, view2)

    def get_matches(self, view1, view2):
        """Extracts feature matches between two views"""
        matches = self.matcher.match(view1.descriptors, view2.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get the keypoints from the matches
        src_pts = np.float32([view1.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([view2.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Use RANSAC to find the fundamental matrix and filter matches
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransacReprojThreshold=0.9, confidence=0.99)
        matches_mask = mask.ravel().tolist()

        # Store inliers only
        self.indices1 = [matches[i].queryIdx for i in range(len(matches)) if matches_mask[i]]
        self.indices2 = [matches[i].trainIdx for i in range(len(matches)) if matches_mask[i]]
        self.distances = [matches[i].distance for i in range(len(matches)) if matches_mask[i]]

        logging.info("Computed matches between view %s and view %s", view1.name, view2.name)

        self.draw_matches()  # Draw matches after computing them


    def draw_matches(self):
        """Draws matches between keypoints of two views and saves the result"""
        keypoints1 = [self.view1.keypoints[i] for i in self.indices1]
        keypoints2 = [self.view2.keypoints[i] for i in self.indices2]

        # Draw matches
        img_matches = cv2.drawMatches(self.view1.image, keypoints1, self.view2.image, keypoints2,
                                      [cv2.DMatch(i, i, self.distances[i]) for i in range(len(self.indices1))],
                                      None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        output_path = os.path.join(self.root_path, 'matches_visualization')
        
        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save the image with matches
        filename1 = os.path.basename(self.image_name1)
        filename2 = os.path.basename(self.image_name2)

        filename = filename1 + '_' + filename2 + '_matches.jpg'
        full_output_path = os.path.join(output_path, filename)
        cv2.imwrite(full_output_path, img_matches)
        logging.info("Matches drawn and saved for view pair %s %s", self.image_name1, self.image_name2)


def create_matches(views):
    """Computes matches between every possible pair of views and stores in a dictionary"""

    matches = {}
    for i in range(len(views) - 1):
        matches[(views[i].name, views[i + 1].name)] = Match(views[i], views[i + 1])

    return matches
