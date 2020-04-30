from keras.models import load_model
import sys
import cv2
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../sitting_posture_project/openpose/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../sitting_posture_project/openpose/x64/Release;' +  dir_path + '/../sitting_posture_project/openpose/bin;'
import pyopenpose as op

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../sitting_posture_project/openpose/models/"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

while True:
    # Process and display images
    datum = op.Datum()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    if np.size(datum.poseKeypoints) != 75 :
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "show your whole body!!!", (10, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        points = datum.poseKeypoints[0, :16, :2]
        points = np.reshape(points, (1, 32))
        if 0 in points[0]:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "can't find whole keypoints!!!", (10, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            model = load_model(r'C:\Users\dusti\sitting_posture_project\sitting_model.h5')
            predictions = model.predict(points)
            if predictions >= 0.5:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'wrong', (10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'right', (10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

