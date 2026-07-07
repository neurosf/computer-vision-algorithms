
import cv2
import numpy as np

img_l = cv2.imread('imgg1.jpg')
imgl = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)

img_r = cv2.imread('imgg2.jpg')
imgr = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kpl, desl = sift.detectAndCompute(imgl,None)
kpr, desr = sift.detectAndCompute(imgr,None)

cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_l,kpl,None))
cv2.waitKey(0)

cv2.imshow('original_image_right_keypoints',cv2.drawKeypoints(img_r,kpr,None))
cv2.waitKey(0)

match = cv2.BFMatcher()
matches = match.knnMatch(desr,desl,k=2)

good = []
for m,n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)

draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)

img3 = cv2.drawMatches(img_r,kpr,img_l,kpl,good,None,**draw_params)
resized_img3 = cv2.resize(img3, None, fx=0.5, fy=0.5)
cv2.imshow("Draw Matches Left Right.jpg", resized_img3)
cv2.waitKey(0)

b = 12
Ox = 560
Oy = 1048
fx = 1543
fy = 1552

points = []

for match in good:
    idx_l = match.trainIdx
    idx_r = match.queryIdx
    
    U_l, V_l = kpl[idx_l].pt
    U_r, V_r = kpr[idx_r].pt
    
    Z = (b * fx) / (U_l - U_r)
    
    X = ((U_l - Ox) * b) / (U_l - U_r)
    Y = ((U_l - Oy) * (b * fx)) / (fy * (U_l - U_r))
    
    points.append({'left': (U_l, V_l), 'right': (U_r, V_r), '3d': (X, Y, Z)})

print("Image Points:")
for point in points:
    left_point = point['left']
    right_point = point['right']
    three_d_point = point['3d']
    
    print("Left Image - X:", left_point[0], "Y:", left_point[1])
    print("Right Image - X:", right_point[0], "Y:", right_point[1])
    print("3D Point - X:", three_d_point[0], "Y:", three_d_point[1], "Z:", three_d_point[2])

point1 = points[0]['3d']
point2 = points[1]['3d']

x1, y1, z1 = point1
x2, y2, z2 = point2

distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

print("Distance between point1 and point2:", distance)