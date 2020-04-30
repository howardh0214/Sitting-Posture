import tensorflow as tf
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

for info in os.listdir('data/testing_set/0'):
    image_path = os.path.join('data/testing_set/0', info)
    pic = cv2.imread(image_path)
    pic = cv2.resize(pic, (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path, pic)

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + 'openpose\\python\\openpose\\Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + 'openpose\\x64\\Release;' +  dir_path + 'openpose\\bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append('openpose\\python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="data/testing_set/0", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "models"

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
imagePaths = op.get_images_on_directory(args[0].image_dir)
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
    model = tf.keras.models.load_model('model_pja.h5')
    predictions = model.predict(points)
    if predictions >= 0.5:
        print("wrong")
    else:
        print("right")

    if not args[0].no_display:
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        key = cv2.waitKey(100000)
        if key == 27 : continue

end = time.time()
print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
