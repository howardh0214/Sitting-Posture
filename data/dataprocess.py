import sys
import cv2
import os
import argparse
import time
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../sitting_posture_project/openpose/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../sitting_posture_project/openpose/x64/Release;' +  dir_path + '/../sitting_posture_project/openpose/bin;'
import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../sitting_posture_project/dataset/0/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=True, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../sitting_posture_project/openpose/models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Read frames on directory
imagePaths = op.get_images_on_directory(args[0].image_dir);
start = time.time()

# Process and display images
for imagePath in imagePaths:
    datum = op.Datum()
    imageToProcess = cv2.imread(imagePath)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    points = datum.poseKeypoints[0,:,:2]
    points = np.reshape(points, (1, 50))
    points = points[0,:32]
    points = np.reshape(points, (1, 32))
    if 0 in points[0]:
        continue
    else:
        points = np.append(points, 0)
        points = np.reshape(points, (1, 33))
        with open('data.csv', 'ab') as f:
            np.savetxt(f, points, fmt='%.2f', delimiter=",")

end = time.time()
print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")