#reference: https://pylessons.com/OpenCV-image-stiching-continue

import cv2
import numpy as np

MIN_MATCH_COUNT = 10

def trim(frame):
    #crop 
    if not np.sum(frame[0]):
        print('1')
        return trim(frame[1:])
    #crop 
    if not np.sum(frame[-1]):
        print('2')
        return trim(frame[0:-1])
    #crop 
    if not np.sum(frame[:,0]):
        print('3')
        return trim(frame[:,1:])
    #crop 
    if not np.sum(frame[:,-1]):
        print('4')
        return trim(frame[:,:-2])
    return frame

def stitch (img_l,imgl,img_r,imgr):
    sift = cv2.SIFT_create()

    kpl, desl = sift.detectAndCompute(imgl,None)
    kpr, desr = sift.detectAndCompute(imgr,None)

    #cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_l,kpl,None))
    #cv2.waitKey(0)

    #cv2.imshow('original_image_right_keypoints',cv2.drawKeypoints(img_r,kpr,None))
    #cv2.waitKey(0)

    match = cv2.BFMatcher()
    matches = match.knnMatch(desr,desl,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0,255,0),singlePointColor=None,flags=2)

    img3 = cv2.drawMatches(img_r,kpr,img_l,kpl,good,None,**draw_params)
    cv2.imshow("Draw Matches Left Right.jpg", img3)
    cv2.waitKey(0)

    M = np.eye(3, 3, dtype=np.float32)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kpr[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpl[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(M)

        h,w = imgr.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
    else:
        print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

    print(img_l.shape[1] + img_r.shape[1], img_l.shape[0])

    dst = cv2.warpPerspective(img_r,M,(img_l.shape[1] + img_r.shape[1], img_l.shape[0]))
    #cv2.imshow("dst warpPerspective", dst)
    #cv2.waitKey(0)

    dst[0:img_l.shape[0],0:img_l.shape[1]] = img_l
    #cv2.imshow("add left image to the warped right.jpg", dst)
    #cv2.waitKey(0)

    #cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    #cv2.waitKey(0)
    #cv2.imwrite("original_image_stitched_crop.jpg", trim(dst))
    return trim(dst)


images = []

for i in range(5):
    img = cv2.imread(f'image{i}.jpg')
    images.append(img)

result = images[0]
for i in range(1,4):
    result = stitch(result,cv2.cvtColor(result,cv2.COLOR_BGR2GRAY),images[i],cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))
    cv2.imshow("result", result)
    cv2.waitKey(0)
