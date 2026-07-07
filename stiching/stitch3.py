import cv2
import numpy as np

images = []
for i in range(5):
    img = cv2.imread(f'image{i}.jpg')
    images.append(img)

sift = cv2.SIFT_create()

keypoints = []
descriptors = []
for img in images:
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

bf = cv2.BFMatcher()

homographies = []

for i in range(len(images) - 1):
    matches = bf.knnMatch(descriptors[i], descriptors[i+1], k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[i+1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    homographies.append(H)

h, w = images[0].shape[:2]
x_max = w
x_min = 0
for H in homographies:
    corners = np.array([[0, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1], [w - 1, 0, 1]])
    transformed_corners = np.dot(H, corners.T).T
    transformed_corners /= transformed_corners[:, 2:]
    x_min = min(x_min, np.min(transformed_corners[:, 0]))
    x_max = max(x_max, np.max(transformed_corners[:, 0]))

out_width = int(x_max - x_min)
out_height = h

stitched_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)

for i in range(len(images)):
    H = np.identity(3) if i == 0 else np.dot(np.array([[1, 0, -x_min], [0, 1, 0], [0, 0, 1]]), homographies[i-1])
    img_warped = cv2.warpPerspective(images[i], H, (out_width, out_height))
    stitched_img += img_warped

cv2.imshow('Stitched Image', stitched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('stitched_image.jpg', stitched_img)