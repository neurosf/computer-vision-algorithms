import cv2
import os

def convert(Object):

    frames_dir = os.path.join("./", Object, "images")
    os.makedirs(frames_dir, exist_ok=True)

    vidcap = cv2.VideoCapture('./Videos/' + Object + '.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 5 == 0: frame_filename = os.path.join(frames_dir, f"frame{count:03d}.jpg")
        cv2.imwrite(frame_filename, image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1