from keras.models import load_model
import sys
import cv2
import os
import numpy as np

def set_params():
    params = dict()
    params["num_gpu"] = -1
    params["model_folder"] = "models"
    params["disable_multi_thread"] = True
    params["fps_max"] = -1
    return params
params = set_params()

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/usr/local/python')
from openpose import pyopenpose as op

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cap = cv2.VideoCapture('data/20200505_225516.m4v')
frame_count = 0


while True:
    # Process and display images
    ret, frame = cap.read()
    if frame_count % 15 == 5:
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        font = cv2.FONT_HERSHEY_SIMPLEX
        if np.size(datum.poseKeypoints) != 75:
            cv2.putText(frame, "Missing Point!", (10, 50), font, 1.25, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            points = datum.poseKeypoints[0, :16, :2]
            points = np.reshape(points, (1, 32))
            
            if np.count_nonzero(points) <= 26:
                cv2.putText(frame, "Missing Point!", (10, 50), font, 1.25, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                points = points / 3
                model = load_model('model_pja.h5')
                predictions = model.predict(points)
                if predictions >= 0.6:
                    cv2.putText(frame, 'Bad', (10, 50), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Good', (10, 50), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('frame', frame)
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

