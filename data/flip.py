import cv2
import os

a = 1
path = r'D:\python\sitting_posture_project\dataset\1'
for image in os.listdir(path):
    images = cv2.imread(os.path.join(path, image))
    h_flip = cv2.flip(images, 1)
    name = str(a) + '.jpg'
    cv2.imwrite(os.path.join(path, name), h_flip)
    a += 1
